"""Pinned ONNX bundle files and checksums; this module never loads a model."""
import json
from pathlib import Path

from .pipeline import file_digest, write_json

MODEL_FILES = (
    "encoder_model.onnx", "encoder_model.onnx.data",
    "decoder_model.onnx", "decoder_with_past_model.onnx", "decoder_shared.onnx.data",
    "config.json", "generation_config.json", "tokenizer_src.json", "tokenizer_tgt.json", "tokenizer_meta.json",
)


def record_bundle(directory, config):
    directory = Path(directory)
    missing = [name for name in MODEL_FILES if not (directory/name).is_file()]
    if missing:
        raise ValueError("Downloaded ONNX bundle is incomplete: " + ", ".join(missing))
    manifest = {"repo":config["model_id"], "revision":config["model_revision"],
                "backend":"onnxruntime", "quantization":"int8_dynamic",
                "files":{name:file_digest(directory/name) for name in MODEL_FILES}}
    write_json(directory/"source-manifest.json",manifest)
    return manifest


def verify_bundle(directory, config):
    directory = Path(directory)
    manifest = json.loads((directory/"source-manifest.json").read_text())
    if manifest.get("repo") != config["model_id"] or manifest.get("revision") != config["model_revision"]:
        raise ValueError("Local ONNX bundle does not match the selected repository/revision")
    if manifest.get("backend") != "onnxruntime" or manifest.get("quantization") != "int8_dynamic":
        raise ValueError("Local bundle is not recorded as the selected INT8 ONNX export")
    if set(manifest.get("files",{})) != set(MODEL_FILES):
        raise ValueError("Manifest must cover every required ONNX graph, external weight file and tokenizer")
    if any(not (directory/name).is_file() or file_digest(directory/name) != manifest["files"][name] for name in MODEL_FILES):
        raise ValueError("Local ONNX bundle file checksum mismatch")
    return manifest
