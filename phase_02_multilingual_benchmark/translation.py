"""Append-only machine output, cached translation units, and resumable execution."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time

from .pipeline import LANGUAGES, PHASE, digest, file_digest, latest_raw, read_jsonl, write_json


class Cache:
    def __init__(self, path, namespace):
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS translations (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.namespace = namespace
        self.hits, self.generated = 0, 0

    def get(self, text, target, backend, attempts=3):
        key = digest([self.namespace, text, target])
        found = self.connection.execute("SELECT value FROM translations WHERE key=?",(key,)).fetchone()
        if found:
            self.hits += 1
            return json.loads(found[0])
        error = None
        for attempt in range(attempts):
            try:
                value = backend.translate(text,target)
                if not value["text"].strip():
                    raise ValueError("Empty translation output")
                self.connection.execute("INSERT INTO translations VALUES (?,?)",(key,json.dumps(value,ensure_ascii=False)))
                self.connection.commit()
                self.generated += 1
                return value
            except Exception as exc:
                error = exc
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Translation failed after {attempts} attempts: {type(error).__name__}: {error}") from error


def run_translation(rows, audit, config, output, backend_factory, resume=False, retry_failed=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest_path = output/"run_manifest.json"
    signature = {"config_sha256":digest(config),"source_file_sha256":audit["file_sha256"],
                 "selected_rows":[{"id":r["id"],"split":r["split"],"source_row_sha256":r["source_row_sha256"]} for r in rows],
                 "pipeline_source_sha256":{p.name:file_digest(p) for p in (PHASE/"pipeline.py",PHASE/"translation.py",PHASE/"indictrans_backend.py",PHASE/"model_artifacts.py")}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume or manifest["signature"] != signature:
            raise ValueError("Existing run requires --resume and identical config, source selection and code; raw outputs were not overwritten")
        if any(not (output/name).exists() or file_digest(output/name) != expected for name,expected in manifest.get("raw_files_sha256",{}).items()):
            raise ValueError("Recorded raw-file hashes changed; preserve and investigate the existing run before resuming")
    else:
        if any(p.name != ".gitkeep" for p in output.iterdir()):
            raise ValueError("Output directory contains untracked/legacy raw files; use a new run directory")
        manifest = {"signature":signature,"status":"initializing","started_utc":datetime.now(timezone.utc).isoformat(),
                    "configuration":config,
                    "invocation_arguments":sys.argv,"python":platform.python_version(),"platform":platform.platform(),
                    "model_id":config["model_id"],"model_revision":config["model_revision"],"provider":config["provider"],
                    "backend":config["backend"],"quantization":config["quantization"],"base_model_id":config["base_model_id"],
                    "chat_prompt":None,"temperature":None,"api_token_consumption":None,"api_cost":None,
                    "expected_translated_qa_rows":len(rows)*4,"generated_rows":0,"human_reviewed_rows":0}
        write_json(manifest_path,manifest)
    started = time.monotonic()
    cache = None
    try:
        backend = backend_factory()
        if resume and manifest.get("runtime") and manifest["runtime"] != backend.metadata:
            raise ValueError("Resume would mix model/runtime versions or devices; start a separate raw run")
        manifest.update({"status":"running","runtime":backend.metadata})
        write_json(manifest_path,manifest)
        namespace = digest([signature["config_sha256"],signature["pipeline_source_sha256"],backend.metadata])
        cache = Cache(output/"translation_cache.sqlite",namespace)
        for language,(name,target,_) in LANGUAGES.items():
            path = output/(name+".jsonl")
            previous = {(r["original_id"],r["language"]):r for r in latest_raw(read_jsonl(path))} if path.exists() else {}
            with path.open("a",encoding="utf-8") as stream:
                for source in rows:
                    old = previous.get((source["id"],language))
                    if old and (old["translation_status"] == "complete" or not retry_failed):
                        continue
                    record = {"original_id":source["id"],"language":language,"split":source["split"],
                              "source_row_sha256":source["source_row_sha256"],"model_id":config["model_id"],
                              "model_revision":config["model_revision"],"target_language_tag":target,
                              "backend":config["backend"],"quantization":config["quantization"],"base_model_id":config["base_model_id"],
                              "attempt":old["attempt"]+1 if old else 1,"created_utc":datetime.now(timezone.utc).isoformat(),
                              "translations":{"context":"","question":"","answers":[]},"field_outputs":{},"generation_issues":[]}
                    try:
                        for field in ("context","question"):
                            value = cache.get(source[field],target,backend,config["runtime"]["retry_attempts"])
                            record["translations"][field] = value["text"]
                            record["field_outputs"][field] = value
                            record["generation_issues"].extend(value["generation_issues"])
                        record["field_outputs"]["answers"] = []
                        for text in source["answers"]["text"]:
                            value = cache.get(text,target,backend,config["runtime"]["retry_attempts"])
                            record["translations"]["answers"].append(value["text"])
                            record["field_outputs"]["answers"].append(value)
                            record["generation_issues"].extend(value["generation_issues"])
                        record["translation_status"] = "complete"
                    except Exception as exc:
                        record.update({"translation_status":"failed","error":str(exc)})
                    record["generation_issues"] = sorted(set(record["generation_issues"]))
                    stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    manifest["generated_rows"] += 1
                    manifest["cache_hits"], manifest["cache_units_generated"] = cache.hits,cache.generated
                    manifest["last_original_id"],manifest["last_language"] = source["id"],language
                    write_json(manifest_path,manifest)
                    if manifest["generated_rows"] % 10 == 0:
                        print(f"Translation: {manifest['generated_rows']} appended rows; latest {language}/{source['id']}",flush=True)
        records = [r for name,_,_ in LANGUAGES.values() for r in latest_raw(read_jsonl(output/(name+".jsonl")))]
        manifest.update({"status":"machine_translation_complete" if all(r["translation_status"] == "complete" for r in records) else "machine_translation_has_failures",
                         "observed_unique_rows":len(records),"successful_unique_rows":sum(r["translation_status"] == "complete" for r in records),
                         "raw_files_sha256":{name+".jsonl":file_digest(output/(name+".jsonl")) for name,_,_ in LANGUAGES.values()}})
    except Exception as exc:
        manifest.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc)})
        raise
    finally:
        manifest["last_session_elapsed_seconds_monotonic"] = time.monotonic()-started
        manifest["total_recorded_execution_seconds"] = manifest.get("total_recorded_execution_seconds",0)+manifest["last_session_elapsed_seconds_monotonic"]
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path,manifest)
        if cache:
            cache.connection.close()
    return manifest
