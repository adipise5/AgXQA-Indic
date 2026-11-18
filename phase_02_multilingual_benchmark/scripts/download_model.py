"""Download the authorized pinned checkpoint and record hashes; no model execution."""
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from phase_02_multilingual_benchmark.pipeline import PHASE, load_config
from phase_02_multilingual_benchmark.model_artifacts import MODEL_FILES, record_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PHASE/"config.json")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    from huggingface_hub import snapshot_download
    path=Path(snapshot_download(config["model_id"],revision=config["model_revision"],local_dir=args.output_dir,
               allow_patterns=list(MODEL_FILES),max_workers=2))
    record_bundle(path,config)
    print({"checkpoint":str(path),"neural_execution":False,"remote_code_executed":False})


if __name__ == "__main__":
    main()
