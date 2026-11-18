"""IndicTrans2 INT8 ONNX backend; sessions load only on explicit model execution."""
import importlib.metadata
import json
from pathlib import Path
import re

from .pipeline import canonical_text
from .model_artifacts import verify_bundle


def chunk_text(text, budget, measure):
    """Keep all normalized source words; never silently truncate long inputs."""
    text = canonical_text(text)
    if not text:
        raise ValueError("Empty translation input")
    if measure(text) <= budget:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    chunks, current = [], ""
    for sentence in sentences:
        combined = (current + " " + sentence).strip()
        if measure(combined) <= budget:
            current = combined
            continue
        if current:
            chunks.append(current)
            current = ""
        if measure(sentence) <= budget:
            current = sentence
            continue
        words = sentence.split()
        while words:
            low, high, take = 1, len(words), 0
            while low <= high:
                middle = (low + high) // 2
                if measure(" ".join(words[:middle])) <= budget:
                    take = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if take == 0:
                raise ValueError("A source word exceeds the input-token budget; manual segmentation required")
            chunks.append(" ".join(words[:take]))
            words = words[take:]
    if current:
        chunks.append(current)
    if " ".join(chunks) != text or any(measure(c) > budget for c in chunks):
        raise ValueError("Chunking lost source text or exceeded the token budget")
    return chunks


def cached_feed(values, layers):
    """The export exposes four KV tensors per layer, in decoder/encoder key/value order."""
    if len(values) != 4*layers:
        raise ValueError("Unexpected decoder KV-cache layout")
    names = ("decoder.key","decoder.value","encoder.key","encoder.value")
    return {f"past_key_values.{layer}.{name}":values[4*layer+offset]
            for layer in range(layers) for offset,name in enumerate(names)}


def greedy_tokens(step, start_id, eos_id, limit):
    """Pure decoding control: start token counts toward the total output budget."""
    if limit < 2:
        raise ValueError("Total target budget must be at least two")
    tokens = [start_id]
    for index in range(limit-1):
        token = step(tokens[-1], index == 0)
        if type(token) is not int or token < 0:
            raise ValueError("Decoder returned an invalid token ID")
        tokens.append(token)
        if token == eos_id:
            return tokens, False
    return tokens, True


class Backend:
    def __init__(self, config, device="cpu", local_model=None):
        if device != "cpu":
            raise ValueError("The selected INT8 ONNX backend uses CPUExecutionProvider")
        if local_model is None:
            raise ValueError("Download the pinned ONNX bundle separately and pass --local-model")
        bundle = verify_bundle(local_model,config)
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            from IndicTransToolkit.IndicTransToolkit import IndicProcessor
        for package,expected in (("onnxruntime",config["onnxruntime_version"]),("indictranstoolkit",config["toolkit_version"])):
            if importlib.metadata.version(package) != expected:
                raise ValueError(f"Install pinned {package}=={expected} before translating")
        self.np, self.config, self.Processor = np, config, IndicProcessor
        directory = Path(local_model)
        self.source_tokenizer = Tokenizer.from_file(str(directory/"tokenizer_src.json"))
        self.target_tokenizer = Tokenizer.from_file(str(directory/"tokenizer_tgt.json"))
        self.source_tokenizer.no_truncation()
        self.source_tokenizer.no_padding()
        self.target_tokenizer.no_truncation()
        self.target_tokenizer.no_padding()
        self.token_meta = json.loads((directory/"tokenizer_meta.json").read_text())
        model = json.loads((directory/"config.json").read_text())
        generation = json.loads((directory/"generation_config.json").read_text())
        self.start_id = int(generation["decoder_start_token_id"])
        self.eos_id = int(generation["eos_token_id"])
        self.layers = model["decoder_layers"]
        if config["generation"]["max_source_tokens"] > model["max_source_positions"] or config["generation"]["max_target_tokens"] > model["max_target_positions"]:
            raise ValueError("Configured token budget exceeds the checkpoint's positional limits")
        options = ort.SessionOptions()
        options.intra_op_num_threads = config["runtime"]["threads"]
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder,self.decoder,self.decoder_past = [ort.InferenceSession(str(directory/name),sess_options=options,
                       providers=["CPUExecutionProvider"]) for name in
                       ("encoder_model.onnx","decoder_model.onnx","decoder_with_past_model.onnx")]
        for session in (self.decoder,self.decoder_past):
            if len(session.get_outputs()) != 1 + 4*self.layers:
                raise ValueError("ONNX decoder output layout differs from the inspected export")
        self.metadata = {"model_id":config["model_id"],"model_revision":config["model_revision"],
            "base_model_id":config["base_model_id"],"backend":"onnxruntime","quantization":"int8_dynamic",
            "device":"cpu","providers":["CPUExecutionProvider"],"decoding":"greedy","batch_size":1,
            "intra_op_threads":options.intra_op_num_threads,"inter_op_threads":1,
            "graph_optimization":"ORT_ENABLE_ALL","trust_remote_code":False,"model_files_sha256":bundle["files"],
            "packages":{name:importlib.metadata.version(name) for name in
                        ("onnxruntime","numpy","tokenizers","indictranstoolkit","huggingface-hub")}}

    def measure(self, text, target):
        probe = self.Processor(inference=True)
        prepared = probe.preprocess_batch([text],src_lang="eng_Latn",tgt_lang=target)[0]
        return len(self.source_tokenizer.encode(prepared).ids)

    def translate(self, text, target):
        cfg, np = self.config, self.np
        chunks = chunk_text(text,cfg["generation"]["max_source_tokens"],lambda value:self.measure(value,target))
        records = []
        for chunk in chunks:
            processor = self.Processor(inference=True)
            prepared = processor.preprocess_batch([chunk],src_lang="eng_Latn",tgt_lang=target)[0]
            encoded = self.source_tokenizer.encode(prepared)
            if len(encoded.ids) > cfg["generation"]["max_source_tokens"]:
                raise ValueError("Input exceeded source budget; no truncation was applied")
            if any(token < 0 or token >= self.token_meta["src_dict_size"] for token in encoded.ids):
                raise ValueError("Source tokenizer produced an out-of-dictionary token; no silent UNK substitution")
            input_ids = np.array([encoded.ids],dtype=np.int64)
            mask = np.array([encoded.attention_mask],dtype=np.int64)
            hidden = self.encoder.run(["last_hidden_state"],{"input_ids":input_ids,"attention_mask":mask})[0]
            past = None

            def step(last_token, first):
                nonlocal past
                feed = {"input_ids":np.array([[last_token]],dtype=np.int64),"encoder_attention_mask":mask}
                if first:
                    feed["encoder_hidden_states"] = hidden
                    outputs = self.decoder.run(None,feed)
                else:
                    outputs = self.decoder_past.run(None,{**feed,**cached_feed(past,self.layers)})
                scores = outputs[0][0,-1,:]
                if not np.isfinite(scores).all():
                    raise ValueError("Non-finite ONNX decoder logits")
                past = list(outputs[1:])
                token = int(np.argmax(scores))
                if token >= self.token_meta["tgt_dict_size"]:
                    raise ValueError("Decoder selected an out-of-dictionary target token")
                return token

            tokens, capped = greedy_tokens(step,self.start_id,self.eos_id,cfg["generation"]["max_target_tokens"])
            decoded = self.target_tokenizer.decode(tokens,skip_special_tokens=True)
            translated = processor.postprocess_batch([decoded],lang=target)[0]
            records.append({"source_text":chunk,"prepared_source_text":prepared,"decoded_text":decoded,
                "generated_token_ids":tokens,"translation_text":translated,"source_token_count":len(encoded.ids),
                "output_token_count":len(tokens),"target_length_limit_reached":capped})
        return {"text":" ".join(row["translation_text"] for row in records),"segments":records,
                "generation_issues":["target_generation_length_limit_reached"] if any(r["target_length_limit_reached"] for r in records) else []}
