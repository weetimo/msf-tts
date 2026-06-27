#!/usr/bin/env python3
"""
ASR Benchmark: Singlish & Singapore languages
Models  : polyglot-lion-0.6b-v1.5, polyglot-lion-1.7b-v1.5, MERaLiON-3-3B-ASR
Datasets: nsc-singlish, librispeech-english, fleurs-malay, fleurs-tamil, fleurs-mandarin
Metrics : WER/CER (%), RTF, latency mean/p50/p95/p99 (s), throughput (xRT)

Usage:
    # Run specific model on specific GPU
    python3 benchmark.py --device 0 --model polyglot-lion-1.7b-v1.5
    python3 benchmark.py --device 1 --model MERaLiON-3-3B-ASR

    # Run all models on default GPU
    python3 benchmark.py

Run in tmux to avoid blocking:
    CUDA_VISIBLE_DEVICES=1 python3 benchmark.py 2>&1 | tee benchmark.log
"""

import gc
import json
import os
import re
import time
import unicodedata
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Must be set before any torch import - use GPU1 (the free one)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch
import jiwer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_SAMPLES = 200      # per dataset
WARMUP_SAMPLES = 3     # discarded before timing
DTYPE = torch.bfloat16
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Parse CLI args for device & model selection
parser = argparse.ArgumentParser(description="ASR Benchmark")
parser.add_argument("--device", type=int, default=None,
                    help="CUDA device index (0, 1, etc). Overrides CUDA_VISIBLE_DEVICES.")
parser.add_argument("--model", type=str, default=None,
                    help="Specific model ID to run (e.g. polyglot-lion-1.7b-v1.5). "
                         "If omitted, runs all models in MODELS list.")
args_parsed, _ = parser.parse_known_args()

# If --device is specified, override CUDA_VISIBLE_DEVICES and DEVICE
if args_parsed.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_parsed.device)
    DEVICE = "cuda:0"  # after remapping, cuda:0 = the specified device
else:
    DEVICE = "cuda:0"

DATASETS = [
    # (hf_id, split, text_col, language_tag, metric)
    ("knoveleng/nsc-singlish",       "test", "text", "singlish", "wer"),
    ("knoveleng/librispeech-english","test", "text", "en",       "wer"),
    ("knoveleng/fleurs-malay",       "test", "text", "ms",       "wer"),
    ("knoveleng/fleurs-tamil",       "test", "text", "ta",       "wer"),
    ("knoveleng/fleurs-mandarin",    "test", "text", "zh",       "cer"),
]

MODELS = [
    # (model_id, backend)  — "qwen" | "meralion"
    ("knoveleng/polyglot-lion-0.6b-v1.5", "qwen"),
    ("knoveleng/polyglot-lion-1.7b-v1.5", "qwen"),
    ("MERaLiON/MERaLiON-3-3B-ASR",        "meralion"),
]

# Filter models based on CLI --model argument
if args_parsed.model:
    MODELS = [(m_id, backend) for m_id, backend in MODELS if args_parsed.model in m_id]
    if not MODELS:
        print(f"[ERROR] No model matched --model={args_parsed.model}. Available: "
              f"polyglot-lion-0.6b-v1.5, polyglot-lion-1.7b-v1.5, MERaLiON-3-3B-ASR")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Text normalization  (matches knoveleng/asr-evalkit TextNormalizer exactly)
# ---------------------------------------------------------------------------

def _fullwidth_to_halfwidth(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


def _remove_punctuation(text: str) -> str:
    cats = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"}
    text = "".join(c for c in text if unicodedata.category(c) not in cats)
    text = re.sub(
        r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~，。、！？；："（）《》【】「」『』—…·]',
        " ", text,
    )
    return text


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _remove_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove spaces between CJK characters
    text = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", text)
    text = _fullwidth_to_halfwidth(text)
    return text.strip()


def compute_metric(predictions: List[str], references: List[str], metric: str) -> float:
    preds_n = [normalize_text(p) for p in predictions]
    refs_n = [normalize_text(r) for r in references]
    # Filter out pairs where reference is empty after normalization
    pairs = [(p, r) for p, r in zip(preds_n, refs_n) if r]
    if not pairs:
        return float("nan")
    preds_f, refs_f = zip(*pairs)
    if metric == "wer":
        return jiwer.wer(list(refs_f), list(preds_f)) * 100
    else:
        return jiwer.cer(list(refs_f), list(preds_f)) * 100


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _find_local_parquet_files(hf_id: str, split: str) -> List[str]:
    """Return local cached parquet file paths for a given split, sorted."""
    import glob
    name = hf_id.split("/")[-1]
    hub = Path.home() / ".cache/huggingface/hub"
    pattern = str(hub / f"datasets--knoveleng--{name}" / "snapshots" / "*" / "data" / f"{split}*.parquet")
    files = sorted(glob.glob(pattern))
    # Exclude incomplete downloads
    files = [f for f in files if not f.endswith(".incomplete")]
    # Also check non-knoveleng repos (e.g. MERaLiON datasets)
    if not files:
        org = hf_id.split("/")[0].lower()
        pattern2 = str(hub / f"datasets--{org}--{name}" / "snapshots" / "*" / "data" / f"{split}*.parquet")
        files = sorted(glob.glob(pattern2))
        files = [f for f in files if not f.endswith(".incomplete")]
    return files


def load_samples(hf_id: str, split: str, text_col: str, n: int) -> List[dict]:
    """Return list of {array, sr, duration, text} dicts."""
    import pyarrow.parquet as pq
    import soundfile as sf
    import io

    print(f"  Loading dataset {hf_id} / {split} (up to {n} samples) ...")

    # Prefer local cached parquet files to avoid streaming network hangs
    local_files = _find_local_parquet_files(hf_id, split)

    if local_files:
        print(f"    Using {len(local_files)} local parquet file(s)")
        samples = []
        for fpath in local_files:
            if len(samples) >= n:
                break
            try:
                table = pq.read_table(fpath)
            except Exception as e:
                print(f"    [warn] can't read {fpath}: {e}")
                continue

            cols = table.schema.names
            text_c = next((c for c in [text_col, "transcription", "sentence", "text"] if c in cols), None)
            if text_c is None:
                print(f"    [warn] no text column in {fpath}, cols={cols}")
                continue

            for i in range(len(table)):
                if len(samples) >= n:
                    break
                row = {c: table[c][i].as_py() for c in cols}
                text = row.get(text_c, "") or ""
                if not text:
                    continue
                # Audio bytes stored as dict with 'bytes' key, or as bytes directly
                audio_val = row.get("audio")
                if audio_val is None:
                    continue
                try:
                    if isinstance(audio_val, dict):
                        audio_bytes = audio_val.get("bytes")
                        if audio_bytes:
                            array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                        else:
                            path = audio_val.get("path", "")
                            if path and os.path.exists(path):
                                array, sr = sf.read(path, dtype="float32")
                            else:
                                continue
                    elif isinstance(audio_val, (bytes, bytearray)):
                        array, sr = sf.read(io.BytesIO(audio_val), dtype="float32")
                    else:
                        continue
                except Exception as e:
                    print(f"    [warn] audio decode error row {i}: {e}")
                    continue

                # Convert to mono float32
                if array.ndim > 1:
                    array = np.mean(array, axis=-1).astype(np.float32)
                array = array.astype(np.float32)

                dur = row.get("duration")
                if dur is None or dur <= 0:
                    dur = len(array) / sr if sr > 0 else 0.0
                dur = float(dur)

                samples.append({"array": array, "sr": sr, "duration": dur, "text": text})

        print(f"  Loaded {len(samples)} samples from local cache.")
        return samples

    # Fallback: HuggingFace streaming (may require download)
    from datasets import load_dataset
    print(f"    No local cache found, streaming from HuggingFace...")
    ds = load_dataset(hf_id, split=split, streaming=True)

    samples = []
    for item in ds:
        if len(samples) >= n:
            break
        audio = item.get("audio")
        text = item.get(text_col) or item.get("transcription") or item.get("sentence") or ""
        if audio is None or not text:
            continue

        try:
            array = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
        except Exception as e:
            print(f"    [warn] audio error: {e}")
            continue

        if array.ndim > 1:
            array = np.mean(array, axis=-1).astype(np.float32)

        dur = item.get("duration")
        if dur is None or dur <= 0:
            dur = len(array) / sr if sr > 0 else 0.0
        dur = float(dur)

        samples.append({"array": array, "sr": sr, "duration": dur, "text": text})

    print(f"  Loaded {len(samples)} samples.")
    return samples


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

def load_qwen_model(model_id: str):
    from qwen_asr import Qwen3ASRModel
    print(f"  Loading Qwen model {model_id} on {DEVICE} ...")
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=DTYPE,
        device_map=DEVICE,
        max_new_tokens=512,
    )
    return model


def infer_qwen(model, array: np.ndarray, sr: int) -> str:
    results = model.transcribe(audio=(array, sr), language=None)
    return results[0].text if results else ""


def load_meralion_model(model_id: str):
    from meralion_3_asr import Meralion3ASR
    print(f"  Loading MERaLiON model {model_id} on {DEVICE} ...")
    # MERaLiON TransformersBackend takes device= and dtype= (not torch_dtype)
    model = Meralion3ASR.from_pretrained(
        model_id,
        backend="transformers",
        device=DEVICE,
        dtype=DTYPE,
    )
    return model


def infer_meralion(model, array: np.ndarray, sr: int) -> str:
    return model.transcribe((array, sr))


# ---------------------------------------------------------------------------
# Latency stats helper
# ---------------------------------------------------------------------------

def latency_stats(times: List[float]) -> dict:
    a = np.array(times)
    return {
        "mean":  float(np.mean(a)),
        "p50":   float(np.percentile(a, 50)),
        "p95":   float(np.percentile(a, 95)),
        "p99":   float(np.percentile(a, 99)),
        "min":   float(np.min(a)),
        "max":   float(np.max(a)),
    }


# ---------------------------------------------------------------------------
# Benchmark one (model, dataset)
# ---------------------------------------------------------------------------

def benchmark_one(
    model,
    infer_fn,
    samples: List[dict],
    metric: str,
    label: str,
) -> dict:
    predictions, references, latencies, audio_durations = [], [], [], []

    for i, s in enumerate(samples):
        array, sr, dur, ref = s["array"], s["sr"], s["duration"], s["text"]

        t0 = time.perf_counter()
        try:
            pred = infer_fn(model, array, sr)
        except Exception as e:
            print(f"    [warn] inference error sample {i}: {e}")
            pred = ""
        elapsed = time.perf_counter() - t0

        if i < WARMUP_SAMPLES:
            continue  # discard warmup timing

        latencies.append(elapsed)
        audio_durations.append(dur)
        predictions.append(pred)
        references.append(ref)

        if (i - WARMUP_SAMPLES) % 20 == 0:
            n_done = i - WARMUP_SAMPLES + 1
            print(f"    [{label}] {n_done}/{len(samples) - WARMUP_SAMPLES} "
                  f"last_lat={elapsed:.2f}s dur={dur:.1f}s")

    if not latencies:
        return {}

    total_audio = sum(audio_durations)
    total_proc = sum(latencies)
    rtf = total_proc / total_audio if total_audio > 0 else float("nan")
    throughput_xrt = total_audio / total_proc if total_proc > 0 else float("nan")

    score = compute_metric(predictions, references, metric)

    result = {
        "n_samples": len(latencies),
        "metric": metric,
        "score": round(score, 2),
        "latency": latency_stats(latencies),
        "rtf": round(rtf, 4),
        "throughput_xRT": round(throughput_xrt, 2),
        "total_audio_sec": round(total_audio, 1),
        "total_proc_sec": round(total_proc, 1),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Print which GPU we're actually using
    actual_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"Using GPU: {actual_gpu} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"CUDA visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Samples/dataset: {MAX_SAMPLES}, warmup: {WARMUP_SAMPLES}\n")

    # Pre-load all datasets once (avoid re-downloading per model)
    print("=== Pre-loading datasets ===")
    dataset_cache: Dict[str, List[dict]] = {}
    for hf_id, split, text_col, lang, metric in DATASETS:
        key = hf_id.split("/")[-1]
        dataset_cache[key] = load_samples(hf_id, split, text_col, MAX_SAMPLES)
    print()

    all_results = {}

    for model_id, backend in MODELS:
        model_name = model_id.split("/")[-1]
        print(f"=== Model: {model_name} (backend={backend}) ===")
        all_results[model_name] = {}

        # Load model
        try:
            if backend == "qwen":
                model = load_qwen_model(model_id)
                infer_fn = infer_qwen
            else:
                model = load_meralion_model(model_id)
                infer_fn = infer_meralion
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            all_results[model_name]["load_error"] = str(e)
            continue

        print(f"  GPU memory after load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")

        for hf_id, split, text_col, lang, metric in DATASETS:
            ds_name = hf_id.split("/")[-1]
            samples = dataset_cache.get(ds_name, [])
            if not samples:
                print(f"  [skip] {ds_name}: no samples loaded")
                continue

            print(f"\n  Dataset: {ds_name} ({lang}, {metric.upper()}, {len(samples)} samples)")
            label = f"{model_name}/{ds_name}"

            result = benchmark_one(model, infer_fn, samples, metric, label)
            all_results[model_name][ds_name] = result

            # Save incrementally
            out_path = RESULTS_DIR / "results.json"
            out_path.write_text(json.dumps(all_results, indent=2))
            if result:
                print(f"  => {metric.upper()}={result['score']:.2f}% "
                      f"RTF={result['rtf']:.3f} "
                      f"lat_mean={result['latency']['mean']:.2f}s "
                      f"p95={result['latency']['p95']:.2f}s "
                      f"xRT={result['throughput_xRT']:.1f}x")

        # Unload model
        print(f"\n  Unloading {model_name} ...")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU memory after unload: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated\n")

    # Save final
    final_path = RESULTS_DIR / "results.json"
    final_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {final_path}")

    # Print summary table
    print_summary(all_results)


def print_summary(results: dict):
    ds_names = [hf_id.split("/")[-1] for hf_id, *_ in DATASETS]
    metrics_map = {hf_id.split("/")[-1]: m for hf_id, _, _, _, m in DATASETS}

    col_w = 28
    header = f"{'Model':<28}" + "".join(f"{n[:16]:>18}" for n in ds_names)
    sep = "-" * len(header)
    print(f"\n{'='*80}")
    print("ACCURACY (WER/CER %)")
    print(sep)
    print(header)
    print(sep)
    for model_name, ds_results in results.items():
        row = f"{model_name:<28}"
        for ds in ds_names:
            r = ds_results.get(ds, {})
            score = r.get("score", float("nan"))
            m = metrics_map.get(ds, "?")
            row += f"{score:>17.2f}%"
        print(row)

    print(f"\n{'='*80}")
    print("SPEED: RTF (lower=faster) | xRT=throughput (higher=faster) | lat_mean (s)")
    print(sep)
    header2 = f"{'Model':<28}" + "".join(f"{n[:10]:>18}" for n in ds_names)
    print(header2)
    print(sep)
    for model_name, ds_results in results.items():
        row_rtf = f"{'  '+model_name+' RTF':<28}"
        row_lat = f"{'  '+model_name+' lat(s)':<28}"
        for ds in ds_names:
            r = ds_results.get(ds, {})
            rtf = r.get("rtf", float("nan"))
            lat = r.get("latency", {}).get("mean", float("nan"))
            row_rtf += f"{rtf:>17.3f} "
            row_lat += f"{lat:>17.2f}s"
        print(row_rtf)
        print(row_lat)
        print()


if __name__ == "__main__":
    main()
