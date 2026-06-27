#!/usr/bin/env python3
"""Quick smoke test: polyglot-lion-1.7b on 2 nsc-singlish samples."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import time

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Free: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB\n")

# Load 2 samples via streaming
from datasets import load_dataset
print("Loading dataset (streaming)...")
ds = load_dataset("knoveleng/nsc-singlish", split="test", streaming=True)
samples = []
for item in ds:
    audio = item["audio"]
    try:
        array = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
    except Exception as e:
        print(f"  skip (audio error): {e}")
        continue
    dur = item.get("duration") or len(array)/sr
    samples.append({"array": array, "sr": sr, "dur": float(dur), "text": item["text"]})
    print(f"  sample {len(samples)}: sr={sr}, dur={float(dur):.2f}s, text={item['text'][:50]}")
    if len(samples) >= 2:
        break
print(f"\nLoaded {len(samples)} samples.\n")

# Load model
from qwen_asr import Qwen3ASRModel
print("Loading polyglot-lion-1.7b-v1.5 (transformers backend)...")
t0 = time.perf_counter()
model = Qwen3ASRModel.from_pretrained(
    "knoveleng/polyglot-lion-1.7b-v1.5",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=512,
)
load_time = time.perf_counter() - t0
print(f"Loaded in {load_time:.1f}s")
print(f"GPU after load: {torch.cuda.memory_allocated(0)/1e9:.2f} GB\n")

# Infer (first is warmup)
for i, s in enumerate(samples):
    t0 = time.perf_counter()
    result = model.transcribe(audio=(s["array"], s["sr"]), language=None)
    elapsed = time.perf_counter() - t0
    pred = result[0].text if result else ""
    warmup = " [warmup]" if i == 0 else ""
    print(f"[{i}] {elapsed:.2f}s{warmup} | ref: {s['text'][:55]}")
    print(f"      pred: {pred[:55]}")
    print()

print("Smoke test PASSED")
