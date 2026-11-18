# Phase 2 — AgXQA-Indic Benchmark Construction

**Owner:** Saunak. **Pilot executed: 18 November 2026.** The user-approved 1B INT8 ONNX pilot completed **20/20 translations in 42.99 seconds**, including loading/checksums. Language-ID checks matched all 20 target languages. Only **2/20 gold answers aligned to a unique exact translated-context span**; 18 need alignment review. Human judgments and final benchmark records remain zero. The full 8,744-record generation run has not started. See [pilot evidence](results/pilot_int8_2026-11-18.json), [ID-level checks](results/pilot_validation.csv) and [completion tracker](results/completion_tracker.json).

The user selected **IndicTrans2 English→Indic 1B, dynamic INT8 ONNX**, replacing the earlier unexecuted 200M plan. Earlier planning notes suggested Gemma/Gemini translations might exist; no such files or provenance were supplied or found in this checkout. Their settings and results are unknown. This implementation records a local ONNX Runtime CPU workflow using the community export `hari31416/indictrans2-en-indic-1B-ONNX-int8` of `ai4bharat/indictrans2-en-indic-1B`. It does not retrospectively label any earlier output.

## 1. Objective

Extend English AgXQA to Hindi, Marathi, Bengali and Tamil while preserving question identity, agricultural meaning, every gold-answer reference and official split membership. Deliver a validated parallel benchmark for extractive QA, with immutable machine output and traceable human corrections. A generated translation is a candidate, not evidence of human validation. The model change is configuration/implementation work; it is not approval to execute inference.

## 2. Original Dataset Description

Source: [`msu-ceco/agxqa_v1`](https://huggingface.co/datasets/msu-ceco/agxqa_v1), pinned to **`9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c`**. The task is English agricultural/irrigation extractive question answering. Each source JSONL record contains `id`, `context`, `question`, parallel `answers.text`/`answers.answer_start` lists, `category` and `references`.

The three `agxqa-{train,validation,test}-2024-06-11.jsonl` files are verified against fixed SHA-256 values before use. [source_preflight.json](results/source_preflight.json) records the hashes and measured audit. The official files are never edited. Source inspection found **391 distinct contexts, 2,185 distinct question strings and 2,045 distinct answer strings** across 2,186 unique question IDs. Repeated text is cached during translation without changing question counts.

## 3. Supported Languages

| Language | Code | IndicTrans2 target tag | Train target | Validation target | Test target | Total target |
|---|---|---|---:|---:|---:|---:|
| English | en | Source `eng_Latn` | 1,503 | 353 | 330 | 2,186 |
| Hindi | hi | `hin_Deva` | 1,503 | 353 | 330 | 2,186 |
| Marathi | mr | `mar_Deva` | 1,503 | 353 | 330 | 2,186 |
| Bengali | bn | `ben_Beng` | 1,503 | 353 | 330 | 2,186 |
| Tamil | ta | `tam_Taml` | 1,503 | 353 | 330 | 2,186 |
| All languages | | | 7,515 | 1,765 | 1,650 | 10,930 |

English counts were verified from downloaded bytes. Indic counts are **targets**, comprising 8,744 translated QA records. The actual pilot contains five validation questions per Indic language (20 translations); no finalized records exist.

## 4. Translation Methodology

The checkpoint is a sequence-to-sequence translation model, accessed through its dynamic INT8 ONNX export with ONNX Runtime and IndicTransToolkit. Source and target tokenizers are loaded from the export's JSON files. Quantization was performed by the community exporter; this phase does not recalibrate or train weights. Its original base-weight revision is not identified, so `base_model_revision` is explicitly null. It uses source/target language conditioning, not a chat prompt. [translation_prompt.txt](prompts/translation_prompt.txt) records the exact planned conditioning procedure; chat prompt, API sampling temperature, paid API token consumption and API cost are not applicable (`null`). The actual pilot took 42.992904708 seconds, including model loading/checksums; this is not a full-dataset runtime estimate.

Translate each context, question and **all** answer strings independently. The same context is translated once per target language/configuration/runtime and reused through a SQLite cache. Sentence segmentation followed by whitespace segmentation keeps actual preprocessed tokenizer inputs within 256 tokens. No input truncation is allowed. This retains normalized source words, but chunk boundaries can reduce discourse coherence; reviewers must inspect full paragraphs. Answers and questions can also be chunked if they exceed the budget.

Each segment records the normalized English text, preprocessed input, decoded model output, generated token IDs, postprocessed output, source/output token counts and length-cap flags. Decoding starts at token 2, generates at most 255 new tokens, stops after a generated EOS (2), and flags the cap if no EOS appears. Greedy decoding follows the inspected exporter helper; the earlier unrun five-beam setting is not used. Out-of-dictionary token IDs and non-finite logits fail explicitly rather than silently substituting text. Joined translations and field-level records are appended to language JSONL files. NFC and whitespace cleanup happen in separate candidates, leaving raw records unchanged. Target generations reaching the cap are flagged for review, rather than assumed complete in meaning.

Answer alignment accepts a **single exact substring match** in the cleaned context. Missing or repeated matches retain their original answer index and all possible positions in `answer_alignment`, and require human correction. The pipeline neither picks an arbitrary repeated occurrence nor appends an answer to a context. Inflection and paraphrase can make independent answer translation difficult; the pilot will measure this limitation before full execution.

Unit translations are retried at most three times, with one- and two-second delays between attempts. Failed QA records remain in raw logs. `--resume` requires identical configuration, input selection, pipeline source hashes and runtime metadata; `--retry-failed` appends a numbered retry. Successful raw records cannot be replaced. A malformed interrupted JSONL tail fails with an error and must be preserved before explicit recovery. Model/runtime changes require a separate raw run.

## 5. Translation Model and Configuration

| Setting | Prepared value |
|---|---|
| Checkpoint | `hari31416/indictrans2-en-indic-1B-ONNX-int8` |
| Base model | `ai4bharat/indictrans2-en-indic-1B`; original base revision unspecified by exporter |
| Export format | Community dynamic INT8 ONNX, including external weight sidecars |
| Immutable revision | `659294a223583c6bc8686facb8f0dc15d593b2d1` |
| Provider | Local ONNX Runtime 1.27.0, CPUExecutionProvider |
| Toolkit | IndicTransToolkit 1.1.1 |
| Decoding | Greedy argmax, `num_beams=1`, one output, `do_sample=false` |
| Input / output cap | 256 actual preprocessed input tokens / 256 total decoder tokens including its start token |
| Runtime | Uploaded INT8 ONNX graphs, 4 intra-op CPU threads, 1 inter-op thread |
| Device | CPU only; no MPS/CUDA/CoreML provider is selected |
| Batch size | 1 segment; encoder, first-step decoder and cached decoder graphs |
| Seed | 42; exact reproducibility across devices is not guaranteed |
| Remote model code | None executed; explicit graph/JSON file allowlist and SHA-256 manifest |

[translation_config.yaml](configs/translation_config.yaml) and [config.json](config.json) contain identical settings. The default CLI uses JSON so non-model checks need only Python's standard library. The separate Python 3.11 environment was installed, `pip check` passed, and actual ONNX/IndicTransToolkit/tokenizer/language-ID integration completed on 20 pilot records. [requirements-lock.txt](requirements-lock.txt) records the resolved environment. The run manifest records actual package versions, device, invocation and source/input/weight hashes. `config.json` is the exact pilot input; its initial preparation-status label is not a live execution tracker. The pilot report records actual execution status. They do not estimate unrecorded costs.

The selected [community export](https://huggingface.co/hari31416/indictrans2-en-indic-1B-ONNX-int8) was public and ungated at preflight; a Hugging Face token is not required for public download. The official base checkpoint is separately gated. The downloader fetches only the pinned ONNX graphs, external weights and required JSON files, never repository Python. Every required file is hashed and verified before session creation. The pinned bundle was downloaded, checksummed and run successfully on this Mac CPU. The ten required files total **1,155,441,455 bytes** (about 1.16 GB); this is disk size, not peak RAM. A macOS arm64 Python 3.11 wheel for the exporter's ONNX Runtime 1.27.0 pin was verified in public package metadata; that wheel requires macOS 14+. Actual runtime integration passed the pilot. Integer quantization operator names were present in the graph files, but no exhaustive tensor-precision or full-precision parity audit was performed. Native translation quality remains unscored.

## 6. Dataset Construction Pipeline

`Pinned English bytes → immutable per-language MT logs → NFC candidates and exact answer alignment → complete automated audit → native-language review and source adjudication → correction revalidation → gated final build → reports and handoff`.

| Script | Behavior |
|---|---|
| `prepare_source.py` | Download/verify pinned source, audit counts, offsets and source overlap |
| `download_model.py` | Download authorized checkpoint and hash files; no neural execution |
| `translate.py` | Print plan by default; inference requires `--run-model` and an explicit raw directory |
| `clean_translations.py` | Make a new candidate version, preserve raw logs, align answers and run QC |
| `validate_schema.py` | Schema, record coverage, IDs and split checks |
| `validate_answer_spans.py` | Exact character offsets and all-reference completeness |
| `validate_language.py` | Report stored script/language-ID assessments from cleaning |
| `check_data_leakage.py` | Audit source overlap and reject changed question split membership |
| `create_review_worksheets.py` | Create seeded/flagged review requests, never completed judgments |
| `merge_languages.py`, `build_dataset.py` | Same full five-language build gate; refuse incomplete/rejected/unreviewed required records |
| `dataset_statistics.py` | Actual counts, quality/coverage reports and optional plots; no neural execution |

The build writes no final files when requirements fail. Raw-output, cleaned-candidate and source hashes bind corrections to the exact reviewed version; changed candidate text invalidates an earlier judgment. Final directories must be new/empty; files are staged and validated before publication. Existing official document overlap is retained and reported. `human_reviewed/*.jsonl` contains only actual reviewed accepted records; unsampled, unflagged train/validation rows may be `automated_accepted`, with `human_reviewed=false`.

## 7. Dataset Schema

Three views are produced from the same accepted records:

- `data/agxqa_indic/final/{train,validation,test}.jsonl`: flattened SQuAD-style records. `id` is `<original_id>_<language>`, `original_id` is the unchanged source ID, `answers` contains parallel `text` and `answer_start` lists, and records include language, split, source-document identity, validation status, `human_reviewed`, source/raw hashes and review provenance.
- `data/agxqa_indic/final/parallel_multilingual.jsonl`: one group per unchanged `original_id`, one official split and a `languages` object containing all five records.
- `data/agxqa_indic/final/shared/{train,validation,test}.jsonl`: compatibility view for this repository's shared validator/evaluator. `id` is the common source ID and `answers` is a list of `{text, answer_start}` objects. Use these files with `python -m shared.validate`.

Do not feed the primary flattened view directly to the shared validator: its answer/ID convention differs. Python character offsets must satisfy `context[start:start + len(text)] == text`; byte offsets are invalid. All source answer references must be represented. The builder blocks unresolved source gold errors rather than silently deleting references.

## 8. Automated Validation

All generated records require schema, expected `(original_id, language)` coverage, duplicate-key, ID alignment, split, source hash, exact nonempty answer span, reference count and Unicode checks. Indic text uses NFC; replacement characters, surrogates and unexpected control characters are flagged, with meaningful joiners preserved. Missing/ambiguous spans are review failures, not fuzzy automatic matches.

Language QC combines expected-script ratios, long English-passage flags and the non-neural character n-gram classifier `langid==1.1.6`, restricted to the five languages. A script check alone cannot distinguish Hindi and Marathi. Classifier scores are not calibrated probabilities and are not semantic evaluation. `--skip-language-id` explicitly flags every candidate for human language verification. `validate_language.py` reads stored assessments; rerun cleaning into a new version to recompute them.

Numbers are compared as multisets after decimal-digit normalization; a small documented unit alias lexicon flags missing units. These conservative heuristics can miss number words or equivalent conversions and can overflag valid forms. Native reviewers must resolve flagged cases explicitly. There is no automatic claim of semantic equivalence, correct agricultural terminology or absence of semantic near duplicates.

## 9. Human Annotation Guidelines

Follow [annotation_guidelines.md](human_review/annotation_guidelines.md). Review context/question meaning, fluency, agricultural terminology, numbers/units and every gold answer. Score the original candidate 1–5 on semantic equivalence, fluency, terminology and answer correctness, then decide Accept / Correct / Reject. Store original outputs separately from corrections, with actual reviewer identity, timestamp, raw hash and rationale.

| Reviewer | Language | Prepared minimum target |
|---|---|---|
| Aditya Saxena | Hindi | 442 questions plus additional flagged questions |
| Aditya Pise | Marathi | Same |
| Saunak | Bengali | Same |
| Dhiraj | Tamil | Same |

Proposed sampling: 5% train (76), 10% validation (36), 100% test (330), seed 42, per language; the percentages are implementation choices, not requirements inferred from the original paper. Every flagged record also requires review. Current actual review coverage is **0%**. The committed target-language CSVs remain templates and the five English source requests remain unjudged. Twenty additional unjudged pilot review requests were generated locally under `outputs/pilot-int8-2026-11-18/human_review/`; the small pilot cohort is requested in full before scale-up. Assignments are not completed annotations. Resolving English source errors requires real source adjudication in addition to Indic review.

## 10. Translation Quality Evaluation

Report successful translated QA records / expected translated QA records as completion rate. Report all-reference valid-span records / observed candidates as raw span validity; failed/missing coverage is separately visible. Human means and Accept/Correct/Reject rates use actual submitted judgments, while coverage counts unique question IDs over expected IDs by language/split. Oversampling flagged records biases these means and must be disclosed.

Cohen's kappa is computed only for explicitly independent reviewers of the same language/examples, per reviewer pair, with at least two paired observations. It is `null` when absent or mathematically undefined. The four assigned people reviewing different languages do not provide inter-annotator agreement. Before/after span comparisons use matched accepted review-resolved IDs, disclose exclusions, and cannot be presented as full-dataset quality. Human error categories are reported separately from automated issue codes. No BLEU/COMET or semantic scores are invented without suitable reference data and an explicit evaluation protocol.

## 11. Dataset Statistics

| Observation | Actual preparation snapshot |
|---|---:|
| Verified source QA records | 2,186 |
| Distinct source contexts | 391 |
| Translated QA records | 20 pilot validation records (5 per Indic language) |
| Completed human judgments | 0 |
| Final multilingual records emitted | 0 |
| Source references with invalid offsets | 9 across 5 test questions |
| Cross-split reference-document groups | 39 |
| Cross-split exact-context groups | 207 |

See [dataset_statistics.json](results/dataset_statistics.json), [translation_report.json](results/translation_report.json) and shared [statistics.json](../data/agxqa_indic/statistics.json). Source/preparation counts must not be relabeled as released final dataset counts. Unmeasured quality means, span rates with no candidates, kappa and execution duration are `null` or absent, not synthetic estimates.

## 12. Error Analysis

The source questions `1170477`, `1174212`, `1183012`, `1181597` and `481` contain invalid answer offsets. `481` includes a placeholder-like gold string. The builder requires explicit English source review and keeps the original files/Phase 1 experiment intact. A corrected Phase 2 English benchmark is a new annotation version, requiring version disclosure in downstream comparisons.

Pilot automated QC found **17 missing exact answer matches, one ambiguous match, three unit-alias flags and two numerical-value flags**. These categories can overlap. [error_analysis.csv](results/error_analysis.csv) includes these measured flags and the five existing English source-offset flags. Human-confirmed translation-error taxonomy counts remain unmeasured. Human terminology, answer mismatch, grammar, meaning change, missing content and numerical errors can be entered in the review CSV's `error_categories` column. Analysis must distinguish alignment failure from incorrect translation and language heuristics from human-confirmed errors.

## 13. Data Leakage Prevention

All five variants of an original question retain its official split. Question IDs crossing splits are rejected. `source_document_id` is derived from source reference URL(s), with citation text as a labeled fallback. These group IDs are not split-prefixed. They are practical citation groups, not a verified bibliography of unique documents; URL-set identity can miss partial overlaps and semantic duplicates.

The official dataset already overlaps across documents and verbatim contexts: **39 citation groups and 207 exact context groups** cross splits. We preserve this for baseline comparability and disclose it, rather than claim document-disjoint evaluation. Shared validation is therefore run one official split at a time. A new document-disjoint benchmark would require a separately versioned experiment and new baselines. Translation caching across splits is deterministic reuse of the same frozen MT output, not training on test answers. Human test review must remain independent of model performance.

## 14. Results and Visualizations

Files under `results/` include the source preflight, per-record automated audit, header-only human evaluation table, actual source issue counts, machine-readable completion tracker and five required plots. The figures now show 2,186 prepared English rows plus five machine candidates per Indic language, actual automated issue counts, and raw answer-span validity. Human quality/review-outcome plots retain missing-observation notices; no after-correction curve exists. No illustrative scores from the brief are used.

Preparation verification: **46 repository tests passed**, including 29 Phase 2 tests using synthetic data/fake translation backends and no neural execution. Shared synthetic validation/evaluation also passed. An actual source-only build preflight exited 2 and emitted no final files, correctly blocking missing translations and required source judgments. See [verification.json](results/verification.json); Actual ONNX graph loading, 20-record MT, and language-classifier integration subsequently passed. Human review, corrections and full generation remain pending.

The actual pilot produced the following observations. A valid exact span does not establish a semantically correct gold answer, and a missing span does not by itself prove a bad translation.

| Language | Completed translations | Unique exact answer spans | Language-ID matches |
|---|---:|---:|---:|
| Hindi | 5/5 | 0/5 | 5/5 |
| Marathi | 5/5 | 1/5 | 5/5 |
| Bengali | 5/5 | 0/5 | 5/5 |
| Tamil | 5/5 | 1/5 | 5/5 |
| Total | 20/20 | 2/20 (10%) | 20/20 |

Source IDs: `1174220`, `1169433`, `1183050`, `1183018`, `1169413`; all belong to validation. Raw outputs, translated question/context previews, environment freeze, logs and unjudged review forms are saved locally under `outputs/pilot-int8-2026-11-18/`. The preview file is `results/pilot_translations.jsonl` within that run directory. Aggregate reports and ID-only validation are committed; raw research text/weights remain local. The main statistics retain full target denominators, while the pilot summary reports completion against the 20 requested records.

The model completed in 42.99 monotonic seconds. The macOS `time -l` wrapper could not collect peak memory because the sandbox denied `kern.clockrate`; the translation itself completed all records and the manifest/output hashes verified. Peak model memory is **unavailable**, not zero or an estimate. A separate timing-tool diagnostic did not rerun the model.

**Do not scale this independent-answer-translation strategy to the full dataset yet.** Exact span validity was only 10% in this small, nonrepresentative pilot. An answer-projection or native human-alignment protocol must be evaluated first; no answers were inserted into contexts and no corrections fabricated. After actual review, regenerate reports from the preserved candidates and completed worksheets. The paired span plot will use the same accepted review-resolved IDs before/after; raw overall candidate validity and unresolved records remain in the report. This preparation is not a completed Phase 2 submission.

## 15. Limitations

The selected INT8 ONNX model ran successfully on CPU for the pilot, but only 2/20 gold answers aligned automatically. Native semantic quality, full-precision parity, full-dataset runtime and peak model RAM remain unmeasured. No repository Python is executed. ONNX sessions and external weight files can use substantially more RAM than the on-disk bundle size. The pilot runtime is measured; quality and generalization require further work before an approved full run. The source base-weight revision is unspecified, and greedy decoding differs from the earlier unrun five-beam proposal. Chunked independent field translation can lose coherence or produce answers that do not match the context; the pilot and native review are essential. Heuristics cannot certify semantics, fluency, terminology or unit equivalence. Original source offsets and document overlap need explicit handling. Human review, multilingual generation, final build, rights/redistribution review and downstream handoff remain pending. Phase 1's English-trained XLM-R model is also still pending; Phase 2 does not satisfy that separate prerequisite for Experiment 3A.

## 16. Reproducibility Instructions

Run from the repository root, using Python 3.11 on this Mac. Keep translation dependencies separate from the Phase 1 environment. The separate environment was installed and validated for this pilot. Recreate its exact resolved Python 3.11/macOS environment with:

```sh
python3.11 -m venv phase_02_multilingual_benchmark/.venv
phase_02_multilingual_benchmark/.venv/bin/python -m pip install -r phase_02_multilingual_benchmark/requirements-lock.txt
mkdir -p phase_02_multilingual_benchmark/outputs
phase_02_multilingual_benchmark/.venv/bin/python -m pip freeze > phase_02_multilingual_benchmark/outputs/environment.freeze.txt
```

`requirements.txt` declares direct dependencies; `requirements-lock.txt` is the actual resolved pilot environment. Preserve the run manifest and environment freeze for later versions. Standard-library synthetic checks do not need translation dependencies:

```sh
python3 -m unittest discover -s tests -v
python3 phase_02_multilingual_benchmark/scripts/prepare_source.py --download
python3 phase_02_multilingual_benchmark/scripts/translate.py --split validation --limit 5
```

The last command prints the plan for **five English validation questions × four languages = 20 translated QA records** and loads no neural model. To reuse already downloaded English files, append `--source-dir /path/to/english-files` to each script. The default local source directory is ignored by Git. This preparation audit used the existing verified English files rather than downloading another dataset copy.

Before model execution, the user must explicitly approve the run. The selected community repository is public/ungated at preflight; token authentication is optional for this download. Download the pinned graph bundle separately without executing repository Python:

```sh
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/download_model.py --output-dir phase_02_multilingual_benchmark/.cache/indictrans2-en-indic-1B-ONNX-int8
```

**Only after explicit approval**, execute the pilot:

```sh
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/translate.py --run-model --split validation --limit 5 --device cpu --local-model phase_02_multilingual_benchmark/.cache/indictrans2-en-indic-1B-ONNX-int8 --output-dir phase_02_multilingual_benchmark/outputs/pilot/raw
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/clean_translations.py --raw-dir phase_02_multilingual_benchmark/outputs/pilot/raw --output-dir phase_02_multilingual_benchmark/outputs/pilot/candidates
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/dataset_statistics.py --raw-dir phase_02_multilingual_benchmark/outputs/pilot/raw --candidates-dir phase_02_multilingual_benchmark/outputs/pilot/candidates --results-dir phase_02_multilingual_benchmark/outputs/pilot/results --plots
```

Inspect success/failure counts, output-length flags, alignment failures, sample fidelity and actual speed/memory. A pilot's 20 target rows are not a complete benchmark. Do not extrapolate quality from five questions. If graph loading or CPU execution fails, inspect the recorded error and revise the plan before further inference; do not mix export/runtime versions in an existing raw run.

The full-run commands below are **prepared and unexecuted**; running all splits needs its own approval after the pilot:

```sh
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/translate.py --run-model --split all --all-rows --device cpu --local-model phase_02_multilingual_benchmark/.cache/indictrans2-en-indic-1B-ONNX-int8 --output-dir data/agxqa_indic/raw_translations
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/clean_translations.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/validate_schema.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/validate_answer_spans.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/validate_language.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/check_data_leakage.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/create_review_worksheets.py --review-dir phase_02_multilingual_benchmark/outputs/review_v1
```

Complete the review CSVs according to the annotation guidelines, preserving raw records. Then:

```sh
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/build_dataset.py --review-dir phase_02_multilingual_benchmark/outputs/review_v1
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/dataset_statistics.py --review-dir phase_02_multilingual_benchmark/outputs/review_v1 --plots --dataset-statistics-output data/agxqa_indic/statistics.json
python3 -m shared.validate data/agxqa_indic/final/shared/train.jsonl
python3 -m shared.validate data/agxqa_indic/final/shared/validation.jsonl
python3 -m shared.validate data/agxqa_indic/final/shared/test.jsonl
```

Validation scripts exit nonzero for missing full-benchmark coverage or failed checks; this is expected on the present incomplete, unreviewed pilot snapshot. Until all required translations, source corrections and reviews exist, the build exits nonzero and writes a blocking report, with no final benchmark. Preserve/version earlier reports before regenerating them for a new candidate/review snapshot. A finished handoff requires final file hashes, license/source documentation, split counts, full review coverage disclosure, benchmark version and downstream owners' acknowledgment. Never commit authentication secrets, caches or large model weights. Raw research data publication requires checking source redistribution terms.

## 17. Contributors

Saunak owns Phase 2, translation methodology, Bengali review and integration. Aditya Saxena is assigned Hindi review, Aditya Pise Marathi review and Dhiraj Tamil review. These assignments come from the project brief; this snapshot does not claim any of them has performed a review. No external messages or reviewer requests were sent.

## 18. References

- [AgXQA dataset, pinned revision](https://huggingface.co/datasets/msu-ceco/agxqa_v1/tree/9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c).
- [Original AgXQA authors' repository](https://github.com/MSU-CECO/agxqa_benchmark_v1).
- [Selected community INT8 ONNX checkpoint](https://huggingface.co/hari31416/indictrans2-en-indic-1B-ONNX-int8/tree/659294a223583c6bc8686facb8f0dc15d593b2d1).
- [Official full-size base checkpoint](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B).
- [Inspected exporter inference interface](https://github.com/Hari31416/indictrans2-onnx-export/blob/c488d66bd1a59bcd419d246e6c2b85c17384d9bc/src/translate.py).
- [IndicTrans2 official inference example, inspected source revision](https://github.com/AI4Bharat/IndicTrans2/blob/4f08e39cc6bf13cd62e2445dc725f22bff1a9219/huggingface_interface/example.py).
- [IndicTransToolkit author repository, inspected source revision](https://github.com/VarunGumma/IndicTransToolkit/tree/3efb8418d0721b4ce267c2b3586899d313191357) and [1.1.1 package](https://pypi.org/project/indictranstoolkit/1.1.1/).
- [langid.py author repository](https://github.com/saffsd/langid.py).
- [Phase 1 measured baseline and source limitations](../phase_01_original_baseline/README.md), [shared evaluation protocol](../docs/evaluation.md), [draft dataset card](../data/agxqa_indic/dataset_card.md).
