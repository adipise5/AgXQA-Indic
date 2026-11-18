"""Phase 2 command implementations. Inference requires an explicit --run-model."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .pipeline import (COUNTS, LANGUAGES, PHASE, load_config, load_source, read_jsonl,
                       read_reviews, resolve_path, source_candidate, write_json, write_jsonl)
from .operations import (assemble, candidate_rows, clean_translations, create_review_worksheets,
                         validate_candidates, write_reports)


def run(command):
    parser = argparse.ArgumentParser(description=f"Phase 2: {command}")
    parser.add_argument("--config",type=Path,default=PHASE/"config.json")
    parser.add_argument("--source-dir",type=Path)
    parser.add_argument("--candidates-dir",type=Path)
    parser.add_argument("--raw-dir",type=Path)
    parser.add_argument("--review-dir",type=Path)
    parser.add_argument("--results-dir",type=Path,default=PHASE/"results")
    if command == "prepare_source":
        parser.add_argument("--download",action="store_true")
        parser.add_argument("--output-dir",type=Path,default=PHASE/"outputs/source")
    if command == "translate":
        parser.add_argument("--run-model",action="store_true",help="Explicitly load and execute the translation checkpoint")
        parser.add_argument("--split",choices=["all",*COUNTS],default="validation")
        parser.add_argument("--limit",type=int,default=5)
        parser.add_argument("--all-rows",action="store_true")
        parser.add_argument("--output-dir",type=Path)
        parser.add_argument("--device",choices=["cpu"],default="cpu")
        parser.add_argument("--local-model",type=Path)
        parser.add_argument("--resume",action="store_true")
        parser.add_argument("--retry-failed",action="store_true")
    if command == "clean_translations":
        parser.add_argument("--output-dir",type=Path)
        parser.add_argument("--skip-language-id",action="store_true",help="Flags every candidate for human language verification")
    if command in ("validate_schema","validate_answer_spans","validate_language"):
        parser.add_argument("--input",type=Path)
    if command in ("merge_languages","build_dataset"):
        parser.add_argument("--output-dir",type=Path)
        parser.add_argument("--human-reviewed-output",type=Path)
    if command == "dataset_statistics":
        parser.add_argument("--plots",action="store_true")
        parser.add_argument("--dataset-statistics-output",type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    source_dir = args.source_dir or resolve_path(config["source_dir"])
    sources, audit = load_source(source_dir,download=command == "prepare_source" and args.download)
    raw_dir = args.raw_dir or resolve_path(config["raw_output_dir"])
    candidates_dir = args.candidates_dir or resolve_path(config["candidate_output_dir"])
    review_dir = args.review_dir or resolve_path(config["review_dir"])
    if command == "prepare_source":
        args.output_dir.mkdir(parents=True,exist_ok=True)
        write_jsonl(args.output_dir/"english_source.jsonl",sources)
        write_json(args.results_dir/"source_preflight.json",{**audit,"translation_model":config["model_id"],
                   "model_revision":config["model_revision"],"target_tags":config["target_language_tags"],
                   "planned_translated_qa_rows":len(sources)*4,"planned_all_language_rows":len(sources)*5,
                   "translation_executed":False,"human_reviews_completed":0})
        print(json.dumps({"source_counts":audit["counts"],"unique_contexts":audit["unique_contexts"],
                          "invalid_source_offsets":len(audit["invalid_source_answer_offsets"]),"neural_execution":False}))
        return
    if command == "translate":
        if args.limit < 1:
            parser.error("--limit must be positive; use --all-rows explicitly for the full selected split(s)")
        selected = [r for r in sources if args.split == "all" or r["split"] == args.split]
        if not args.all_rows:
            selected = selected[:args.limit]
        plan = {"model":config["model_id"],"model_revision":config["model_revision"],"source_rows":len(selected),
                "translated_rows":len(selected)*4,"splits":dict(Counter(r["split"] for r in selected)),
                "languages":config["target_language_tags"],"generation":config["generation"],
                "backend":config["backend"],"quantization":config["quantization"],"device":args.device,
                "model_executed":False,"access_requirement":"The selected community ONNX repository is public and ungated at preflight. Download its pinned bundle separately and pass --local-model; no token is required for public access."}
        if not args.run_model:
            print(json.dumps(plan,indent=2))
            return
        if not args.output_dir:
            parser.error("--run-model requires an explicit fresh --output-dir; raw outputs are never overwritten")
        if not args.local_model:
            parser.error("--run-model requires --local-model from scripts/download_model.py; checkpoint download is a separate step")
        from .indictrans_backend import Backend
        from .translation import run_translation
        manifest = run_translation(selected,audit,config,args.output_dir,
                       lambda:Backend(config,args.device,args.local_model),resume=args.resume,retry_failed=args.retry_failed)
        print(json.dumps({"status":manifest["status"],"observed_unique_rows":manifest["observed_unique_rows"],
                          "successful_unique_rows":manifest["successful_unique_rows"]},indent=2))
        return
    if command == "clean_translations":
        if not (raw_dir/"run_manifest.json").exists():
            raise ValueError("No recorded raw translation run found. Existing unverified legacy files must be imported with their actual provenance first.")
        count = clean_translations(sources,raw_dir,args.output_dir or candidates_dir,use_langid=not args.skip_language_id)
        print(json.dumps({"candidates":count,"human_reviewed":False,"raw_outputs_modified":False}))
        return
    candidates = candidate_rows(candidates_dir)
    if not candidates:
        candidates = [source_candidate(row) for row in sources]
    reviews = read_reviews(review_dir)
    raw_manifest = json.loads((raw_dir/"run_manifest.json").read_text()) if (raw_dir/"run_manifest.json").exists() else None
    if command in ("validate_schema","validate_answer_spans","validate_language"):
        if args.input:
            candidates = read_jsonl(args.input)
        reports,coverage = validate_candidates(candidates,sources)
        write_json(args.results_dir/(command+".json"),{"checks":reports,"coverage":coverage,
                   "language_check_note":"Stored language-ID and script assessments are heuristics; Hindi and Marathi share Devanagari."})
        print(json.dumps({"observed":len(candidates),"missing_pairs":coverage["missing_pair_count"],"valid_span_rows":sum(r["answer_span_valid"] for r in reports)}))
        if command == "validate_schema":
            failed = any(not r["schema_valid"] or any(code in r["issues"] for code in ("duplicate_original_id_language","source_identity_mismatch","unknown_id_or_split_changed")) for r in reports)
        elif command == "validate_answer_spans":
            failed = any(not r["all_reference_spans_valid"] for r in reports)
        else:
            failed = any(r["language"] != "en" and (r["language_id"] != r["language"] or any(code in r["issues"] for code in ("low_expected_script_ratio","unicode_corruption","long_untranslated_latin_passage"))) for r in reports)
        if failed or coverage["missing_pair_count"] or coverage["extra_pairs"]:
            parser.exit(2,"Validation incomplete/failed; inspect the saved per-record report and coverage.\n")
        return
    if command == "check_data_leakage":
        from .pipeline import leakage_report
        mismatches = [r["id"] for r in candidates if r["original_id"] not in {s["id"] for s in sources} or
                      next(s["split"] for s in sources if s["id"] == r["original_id"]) != r["split"]]
        report = {"official_source":leakage_report(sources),"candidate_id_split_mismatches":mismatches,
                  "official_splits_reshuffled":False,"note":"Existing official source-document overlap is retained and disclosed."}
        write_json(args.results_dir/"leakage_report.json",report)
        print(json.dumps(report,indent=2))
        if mismatches:
            raise ValueError("Candidate split membership differs from the official source")
        return
    if command == "create_review_worksheets":
        create_review_worksheets(sources,candidates,config,review_dir)
        print(json.dumps({"review_directory":str(review_dir),"filled_review_judgments":0,"note":"Worksheets are requests, not completed human reviews"}))
        return
    if command in ("merge_languages","build_dataset"):
        report = assemble(sources,candidates,reviews,config,args.output_dir or resolve_path(config["final_output_dir"]),
                          args.human_reviewed_output or resolve_path(config["human_reviewed_dir"]))
        write_json(args.results_dir/"build_report.json",report)
        print(json.dumps({key:value for key,value in report.items() if key not in ("blocked",)},indent=2))
        if report["status"] != "ready":
            parser.exit(2,"Build blocked: required translations, valid spans, or human judgments are missing. No final benchmark was emitted.\n")
        return
    if command == "dataset_statistics":
        report = write_reports(sources,candidates,reviews,config,args.results_dir,raw_manifest)
        if args.dataset_statistics_output:
            write_json(args.dataset_statistics_output,report)
        if args.plots:
            from .visualizations import generate_plots
            generate_plots(report,args.results_dir/"plots")
        print(json.dumps({"status":report["status"],"human_reviews":sum(v["review_judgments"] for v in report["human_quality"].values()),
                          "reported_ratings_are_measured":True,"unmeasured_values_are_null":True}))
        return
    raise ValueError("Unknown Phase 2 operation")
