"""Synthetic pipeline invariants only: no neural/tokenizer/language model is loaded."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from phase_02_multilingual_benchmark import pipeline as p
from phase_02_multilingual_benchmark.indictrans_backend import Backend, chunk_text, cached_feed, greedy_tokens
from phase_02_multilingual_benchmark.model_artifacts import MODEL_FILES, record_bundle, verify_bundle
from phase_02_multilingual_benchmark.operations import assemble, statistics, validate_candidates, clean_translations, candidate_rows
from phase_02_multilingual_benchmark.translation import run_translation


def source(identifier="toy", split="test", question="Where is the water?"):
    row = {"id":identifier,"original_id":identifier,"split":split,"context":"Water reaches roots.",
           "question":question,"answers":{"text":["roots"],"answer_start":[14]},
           "source_document_id":"fixture-document-"+split,"source_document_id_kind":"synthetic_fixture",
           "source_invalid_answer_indices":[]}
    row["source_row_sha256"] = p.digest(row)
    return row


def candidate(row, language="bn"):
    result=p.source_candidate(row)
    result.update(id=row["id"]+"_"+language,language=language,context="জল মূলে যায়।",question="জল কোথায় যায়?",
                  answers={"text":["মূলে"],"answer_start":[3]},translation_status="complete",
                  raw_record_sha256=p.digest([row["id"],language]),automated_issues=[])
    result["candidate_sha256"]=p.candidate_fingerprint(result)
    return result


def judgment(row, status="accept", reviewer="Synthetic reviewer A"):
    row["candidate_sha256"]=p.candidate_fingerprint(row)
    result={field:"" for field in p.REVIEW_FIELDS}
    result.update(question_id=row["original_id"],language=row["language"],split=row["split"],reviewer=reviewer,
                  status=status,reviewed_at="2026-11-18T10:00:00+05:30",raw_record_sha256=row["raw_record_sha256"])
    result["candidate_sha256"]=row["candidate_sha256"]
    result.update({field:"5" for field in p.RATING_FIELDS})
    return result


class AlignmentTests(unittest.TestCase):
    def test_unique_alignment_and_unicode_offsets(self):
        context="জল মূলে যায়।"
        answers,alignment,issues=p.align_answers(context,["  মূলে  "])
        self.assertEqual(answers,[{"text":"মূলে","answer_start":3}])
        self.assertFalse(issues)
        self.assertEqual(alignment[0]["source_answer_index"],0)
        self.assertTrue(p.span_valid(context,"মূলে",3))
        self.assertFalse(p.span_valid(context,"মূলে",True))

    def test_ambiguous_or_missing_answers_are_not_injected(self):
        context="root root"
        answers,alignment,issues=p.align_answers(context,["root","absent"])
        self.assertEqual(answers,[])
        self.assertEqual(alignment[0]["candidate_starts"],[0,5])
        self.assertEqual(set(issues),{"answer_ambiguous_multiple_matches","answer_not_found"})
        self.assertEqual(context,"root root")

    def test_numeric_and_unit_heuristics(self):
        self.assertEqual(p.numeric_values("1,500 litres, 25mm and 3.5%"),p.numeric_values("১,৫০০ লিটার, ২৫মিমি এবং ৩.৫%"))
        self.assertEqual(p.units("25mm and 3.5%"),{"millimetre","percent"})
        self.assertNotEqual(p.numeric_values("25"),p.numeric_values("250"))

    def test_script_and_unicode_checks_have_hindi_marathi_limit(self):
        self.assertFalse(p.script_report("पाणी मुळांपर्यंत पोहोचते","hi")["issues"])
        self.assertFalse(p.script_report("पाणी मुळांपर्यंत पोहोचते","mr")["issues"])
        self.assertIn("unicode_corruption",p.script_report("জল\ufffd","bn")["issues"])
        self.assertIn("low_expected_script_ratio",p.script_report("Water reaches roots","ta")["issues"])

    def test_chunking_retains_all_words_and_enforces_budget(self):
        text="One two three. Four five six seven eight nine. Ten eleven."
        chunks=chunk_text(text,4,lambda value:len(value.split()))
        self.assertEqual(" ".join(chunks),text)
        self.assertTrue(all(len(c.split())<=4 for c in chunks))
        with self.assertRaisesRegex(ValueError,"word exceeds"):
            chunk_text("oversized",3,len)

    def test_malformed_answers_fail_without_crashing(self):
        row=candidate(source())
        for answers in (None,[],{"text":None,"answer_start":None},{"text":["মূলে"],"answer_start":[True]}):
            changed={**row,"answers":answers}
            self.assertTrue(p.schema_issues(changed,require_answer=True))
            self.assertTrue(p.answer_issues(changed))

    def test_raw_retry_order_and_success_immutability(self):
        failed={"original_id":"toy","language":"bn","attempt":1,"translation_status":"failed"}
        success={**failed,"attempt":2,"translation_status":"complete"}
        self.assertEqual(p.latest_raw([failed,success]),[success])
        with self.assertRaises(ValueError):
            p.latest_raw([success,{**success,"attempt":3}])
        with self.assertRaises(ValueError):
            p.latest_raw([success,failed])


class ReviewTests(unittest.TestCase):
    def test_test_split_review_is_mandatory_in_all_languages(self):
        rows=[source("test"),source("train","train")]
        selected=p.review_sample_ids(rows,p.load_config())
        self.assertTrue(all(("test",language) in selected for language in p.LANGUAGES))
        self.assertEqual(selected,p.review_sample_ids(rows,p.load_config()))

    def test_unfilled_worksheets_do_not_count_as_human_reviews(self):
        row=candidate(source())
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"bengali_review.csv"
            with path.open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=p.REVIEW_FIELDS)
                writer.writeheader();writer.writerow({"question_id":"toy","language":"bn"})
            self.assertFalse(p.read_reviews(temporary))
            review=judgment(row);review["semantic_equivalence"]="6"
            with path.open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=p.REVIEW_FIELDS)
                writer.writeheader();writer.writerow(review)
            with self.assertRaisesRegex(ValueError,"1–5"):
                p.read_reviews(temporary)

    def test_hash_and_split_bind_review_to_original_raw_output(self):
        row=candidate(source())
        review=judgment(row);review["raw_record_sha256"]="0"*64
        self.assertEqual(p.apply_review(row,[review])[1],["review_raw_hash_or_split_mismatch"])

    def test_cleaned_candidate_changes_invalidate_prior_review(self):
        row=candidate(source());review=judgment(row)
        row["question"]="জল কেন যায়?"
        self.assertEqual(p.apply_review(row,[review])[1],["review_candidate_hash_mismatch"])

    def test_correction_requires_changes_and_valid_spans(self):
        row=candidate(source())
        review=judgment(row,"correct");review["comments"]="Synthetic correction rationale."
        self.assertIn("correct_decision_contains_no_changes",p.apply_review(row,[review])[1])
        review["corrected_answers_json"]=json.dumps({"text":["মূলে"],"answer_start":[4]})
        self.assertIn("answer_span_mismatch",p.apply_review(row,[review])[1])

    def test_flags_need_explicit_resolution_and_corrected_numbers_are_checked(self):
        row=candidate(source())
        row["automated_issues"]=["language_id_not_run"]
        review=judgment(row)
        self.assertTrue(p.apply_review(row,[review])[1])
        review.update(override_issues="language_id_not_run",comments="Synthetic native-language verification.")
        accepted,issues=p.apply_review(row,[review])
        self.assertFalse(issues);self.assertTrue(accepted["human_reviewed"])
        review.update(status="correct",corrected_context="জল মূলে যায়। 250")
        self.assertIn("context_numeric_values_changed",";".join(p.apply_review(row,[review],source())[1]))

    def test_reviewer_disagreement_requires_named_adjudication(self):
        row=candidate(source())
        first=judgment(row);second=judgment(row,"reject","Synthetic reviewer B")
        self.assertEqual(p.apply_review(row,[first,second])[1],["reviewer_disagreement_requires_adjudication"])
        third=judgment(row,reviewer="Synthetic adjudicator");third["adjudicator"]="Synthetic adjudicator"
        self.assertFalse(p.apply_review(row,[first,second,third])[1])

    def test_missing_translation_or_review_blocks_final_output(self):
        row=source()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"final"
            report=assemble([row],[p.source_candidate(row)],{},p.load_config(),output)
            self.assertEqual(report["status"],"blocked_pending_validation_or_review")
            self.assertFalse(output.exists())

    def test_complete_synthetic_group_writes_both_schemas_and_parallel_view(self):
        sources=[source("toy-"+split,split) for split in p.COUNTS]
        candidates=[item for row in sources for item in [p.source_candidate(row)]+[candidate(row,l) for l in p.LANGUAGES]]
        reviews={}
        # Supply language-specific fixture text so correction revalidation can check script.
        fixtures={"hi":("जल जड़ों तक जाता है।","जल कहाँ जाता है?","जड़ों"),
                  "mr":("पाणी मुळांपर्यंत पोहोचते.","पाणी कुठे पोहोचते?","मुळांपर्यंत"),
                  "bn":("জল মূলে যায়।","জল কোথায় যায়?","মূলে"),
                  "ta":("நீர் வேர்களை அடைகிறது.","நீர் எங்கு செல்கிறது?","வேர்களை")}
        for item in [c for c in candidates if c["language"] != "en"]:
            context,question,answer=fixtures[item["language"]]
            item.update(context=context,question=question,answers={"text":[answer],"answer_start":[context.index(answer)]})
            reviews[(item["original_id"],item["language"])]=[judgment(item)]
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"final"
            report=assemble(sources,candidates,reviews,p.load_config(),output)
            self.assertEqual(report["status"],"ready")
            self.assertEqual(len(p.read_jsonl(output/"test.jsonl")),5)
            self.assertEqual({r["id"] for r in p.read_jsonl(output/"shared/test.jsonl")},{"toy-test"})
            parallel=p.read_jsonl(output/"parallel_multilingual.jsonl")
            self.assertEqual(set(parallel[0]["languages"]),{"en",*p.LANGUAGES})
            with self.assertRaisesRegex(ValueError,"nonempty"):
                assemble(sources,candidates,reviews,p.load_config(),output)

    def test_unmeasured_ratings_and_agreement_are_null(self):
        row=source()
        report=statistics([row],[p.source_candidate(row)],{},p.load_config())
        self.assertIsNone(report["human_quality"]["bn"]["fluency"])
        self.assertIsNone(report["cohens_kappa"])
        self.assertEqual(report["observed"]["bn/test"]["translation_completion_rate"],0)
        self.assertIsNone(report["observed"]["bn/test"]["span_validity_rate"])

    def test_kappa_requires_explicit_independent_same_example_pairs(self):
        rows=[source("a"),source("b")]
        candidates=[candidate(row) for row in rows]
        reviews={(r["original_id"],"bn"):[judgment(r),judgment(r,"reject","Synthetic reviewer B")] for r in candidates}
        self.assertIsNone(statistics(rows,candidates,reviews,p.load_config())["cohens_kappa"])
        for entries in reviews.values():
            for review in entries:
                review["independent_review"]="true"
        reviews[("b","bn")][1]["status"]="accept"
        result=statistics(rows,candidates,reviews,p.load_config())["cohens_kappa"]
        self.assertEqual(result[0]["paired_examples"],2)
        self.assertEqual(result[0]["cohens_kappa"],0)


class ONNXBackendTests(unittest.TestCase):
    def test_selected_configuration_is_pinned_int8_cpu(self):
        config=p.load_config()
        self.assertEqual(config["model_id"],"hari31416/indictrans2-en-indic-1B-ONNX-int8")
        self.assertEqual(config["model_revision"],"659294a223583c6bc8686facb8f0dc15d593b2d1")
        self.assertEqual(config["runtime"]["device"],"cpu")
        self.assertEqual(config["generation"]["num_beams"],1)

    def test_greedy_decoder_handles_start_equal_to_eos_and_cached_steps(self):
        calls=[]
        def step(previous,first):
            calls.append((previous,first))
            return 10 if first else 2
        tokens,capped=greedy_tokens(step,2,2,256)
        self.assertEqual(tokens,[2,10,2])
        self.assertFalse(capped)
        self.assertEqual(calls,[(2,True),(10,False)])

    def test_greedy_decoder_respects_total_length_and_flags_missing_eos(self):
        tokens,capped=greedy_tokens(lambda previous,first:10,2,2,4)
        self.assertEqual(tokens,[2,10,10,10])
        self.assertTrue(capped)
        with self.assertRaises(ValueError):
            greedy_tokens(lambda previous,first:-1,2,2,4)

    def test_decoder_cache_maps_all_four_tensors_per_layer(self):
        values=list(range(8))
        feed=cached_feed(values,2)
        self.assertEqual(feed["past_key_values.0.decoder.key"],0)
        self.assertEqual(feed["past_key_values.1.encoder.value"],7)
        self.assertEqual(len(feed),8)
        with self.assertRaisesRegex(ValueError,"cache layout"):
            cached_feed(values[:6],2)

    def test_bundle_manifest_requires_external_weights_and_detects_changes(self):
        config=p.load_config()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for name in MODEL_FILES:
                (directory/name).write_bytes(b"SYNTHETIC FILE, NOT MODEL WEIGHTS")
            manifest=record_bundle(directory,config)
            self.assertEqual(verify_bundle(directory,config),manifest)
            (directory/"decoder_shared.onnx.data").write_bytes(b"SYNTHETIC CORRUPTION")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                verify_bundle(directory,config)

    def test_incomplete_or_wrong_model_bundles_are_rejected(self):
        config=p.load_config()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,"incomplete"):
                record_bundle(temporary,config)
            p.write_json(Path(temporary)/"source-manifest.json",{"repo":"wrong","revision":config["model_revision"]})
            with self.assertRaisesRegex(ValueError,"repository/revision"):
                verify_bundle(temporary,config)

    def test_cpu_and_explicit_local_bundle_are_required_before_loading(self):
        with self.assertRaisesRegex(ValueError,"CPUExecutionProvider"):
            Backend(p.load_config(),device="mps")
        with self.assertRaisesRegex(ValueError,"separately"):
            Backend(p.load_config())
        self.assertIn("encoder_model.onnx.data",MODEL_FILES)
        self.assertFalse(any(name.endswith(".py") for name in MODEL_FILES))


class RawRunTests(unittest.TestCase):
    def test_cached_outputs_resume_without_overwriting_and_changed_config_fails(self):
        class FakeBackend:
            metadata={"fixture_only":True}
            def __init__(self):
                self.calls=[]
            def translate(self,text,target):
                self.calls.append((text,target))
                return {"text":"SYNTHETIC "+text,"segments":[],"generation_issues":[]}
        backend=FakeBackend()
        config=copy.deepcopy(p.load_config());config["runtime"]["retry_attempts"]=1
        rows=[source("a",question="First question?"),source("b",question="Second question?")]
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"raw"
            audit={"file_sha256":{"fixture":"synthetic"}}
            result=run_translation(rows,audit,config,output,lambda:backend)
            self.assertEqual(result["successful_unique_rows"],8)
            self.assertEqual(len(backend.calls),16)  # Four distinct units per language.
            hashes=result["raw_files_sha256"]
            resumed=run_translation(rows,audit,config,output,lambda:backend,resume=True)
            self.assertEqual(resumed["raw_files_sha256"],hashes)
            self.assertEqual(len(backend.calls),16)
            cleaned=Path(temporary)/"candidates"
            self.assertEqual(clean_translations(rows,output,cleaned,use_langid=False),8)
            self.assertEqual(len(candidate_rows(cleaned)),10)
            self.assertTrue(all(not r["human_reviewed"] for r in candidate_rows(cleaned)))
            self.assertEqual({name:p.file_digest(output/name) for name in hashes},hashes)
            changed=copy.deepcopy(config);changed["seed"]=43
            with self.assertRaisesRegex(ValueError,"identical config"):
                run_translation(rows,audit,changed,output,lambda:backend,resume=True)
            with (output/"bengali.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError,"hashes changed"):
                run_translation(rows,audit,config,output,lambda:backend,resume=True)

    def test_failed_records_are_preserved_and_retry_appends_a_new_attempt(self):
        class FakeBackend:
            metadata={"fixture_only":True}
            fail=True
            def translate(self,text,target):
                if self.fail and text.endswith("?"):
                    raise ValueError("Synthetic deliberate failure")
                return {"text":"SYNTHETIC "+text,"segments":[],"generation_issues":[]}
        backend=FakeBackend()
        config=copy.deepcopy(p.load_config());config["runtime"]["retry_attempts"]=1
        with tempfile.TemporaryDirectory() as temporary:
            audit={"file_sha256":{"fixture":"synthetic"}}
            first=run_translation([source()],audit,config,temporary,lambda:backend)
            self.assertEqual(first["successful_unique_rows"],0)
            before=(Path(temporary)/"bengali.jsonl").read_text()
            backend.fail=False
            second=run_translation([source()],audit,config,temporary,lambda:backend,resume=True,retry_failed=True)
            self.assertEqual(second["successful_unique_rows"],4)
            self.assertTrue((Path(temporary)/"bengali.jsonl").read_text().startswith(before))
            rows=p.read_jsonl(Path(temporary)/"bengali.jsonl")
            self.assertEqual([r["attempt"] for r in rows],[1,2])
            self.assertEqual(p.latest_raw(rows)[0]["translation_status"],"complete")

    def test_split_mismatch_and_missing_coverage_are_reported(self):
        row=source()
        item=candidate(row);item["split"]="train"
        checks,coverage=validate_candidates([item],[row])
        self.assertIn("unknown_id_or_split_changed",checks[0]["issues"])
        self.assertEqual(coverage["missing_pair_count"],4)

    def test_schema_validator_reports_malformed_json_objects_and_arrays(self):
        row=source()
        checks,coverage=validate_candidates([[],None,{"original_id":[],"language":{},"answers":{"text":None},"automated_issues":None}],[row])
        self.assertTrue(all(not entry["schema_valid"] for entry in checks))
        self.assertEqual(coverage["missing_pair_count"],5)


if __name__ == "__main__":
    unittest.main()
