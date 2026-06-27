#!/usr/bin/env python3
"""
Download only what's needed for the benchmark:
- Test parquet files for each dataset (single file each, no full-split build)
- Model weights for all 3 models
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from huggingface_hub import hf_hub_download, snapshot_download

DATASETS = [
    ("knoveleng/nsc-singlish",        "data/test-00001-of-00002.parquet"),  # 00000 already cached
    ("knoveleng/librispeech-english",  "data/test-00000-of-00001.parquet"),
    ("knoveleng/fleurs-malay",         "data/test-00000-of-00001.parquet"),
    ("knoveleng/fleurs-tamil",         "data/test-00000-of-00001.parquet"),
    ("knoveleng/fleurs-mandarin",      "data/test-00000-of-00001.parquet"),
]

MODELS = [
    "knoveleng/polyglot-lion-1.7b-v1.5",  # 0.6b already cached
    "MERaLiON/MERaLiON-3-3B-ASR",
]

print("=== Downloading dataset test files ===")
for repo_id, filename in DATASETS:
    print(f"  {repo_id} / {filename} ...")
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        import os as _os
        size_mb = _os.path.getsize(path) / 1e6
        print(f"    OK: {path} ({size_mb:.0f} MB)")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\n=== Downloading model weights ===")
for model_id in MODELS:
    print(f"  {model_id} ...")
    try:
        path = snapshot_download(model_id)
        print(f"    OK: {path}")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\nDone.")
