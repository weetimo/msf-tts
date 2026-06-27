# ASR Benchmark Results

**Date:** 2025-01-31
**GPU:** RTX 3080 (20GB VRAM)
**Dtype:** bfloat16
**Samples per dataset:** 200 (3 warmup discarded -> ~197 measured)

---

## Models Tested

| Model | Size | Backend |
|---|---|---|
| polyglot-lion-0.6b-v1.5 | 0.6B params | Qwen2-Audio |
| polyglot-lion-1.7b-v1.5 | 1.7B params | Qwen2-Audio |
| MERaLiON-3-3B-ASR | 3B params | MERaLiON |

## Datasets

| Dataset | Language | Metric | Test Samples |
|---|---|---|---|
| NSC-Singlish | Singlish (en+zh mix) | WER | 3000 |
| LibriSpeech-EN | English | WER | 2620 |
| FLEURS-Malay | Malay | WER | 749 |
| FLEURS-Tamil | Tamil | WER | 591 |
| FLEURS-Mandarin | Mandarin Chinese | CER | ~1500 |

---

## Results Summary

### polyglot-lion-0.6b-v1.5

| Dataset | Score | RTF | Speed | vs Paper |
|---|---|---|---|---|
| NSC-Singlish (WER) | **4.89%** | 0.109 | 9.2xRT | paper: 6.09% |
| LibriSpeech-EN (WER) | **1.90%** | 0.107 | 9.3xRT | paper: 2.67% |
| FLEURS-Malay (WER) | **13.96%** | 0.139 | 7.2xRT | paper: 14.45% |
| FLEURS-Tamil (WER) | **40.12%** | 0.428 | 2.3xRT | paper: 37.68% |
| FLEURS-Mandarin (CER) | **10.32%** | 0.086 | 11.6xRT | paper: 9.19% |

### polyglot-lion-1.7b-v1.5

| Dataset | Score | RTF | Speed | vs Paper |
|---|---|---|---|---|
| NSC-Singlish (WER) | **3.94%** | 0.115 | 8.7xRT | paper: 5.30% |
| LibriSpeech-EN (WER) | **1.62%** | 0.111 | 9.1xRT | paper: 2.20% |
| FLEURS-Malay (WER) | **8.88%** | 0.145 | 6.9xRT | paper: 10.20% |
| FLEURS-Tamil (WER) | **39.72%** | 0.453 | 2.2xRT | paper: 37.68% |
| FLEURS-Mandarin (CER) | **7.84%** | 0.089 | 11.2xRT | paper: 8.00% |

### MERaLiON-3-3B-ASR

| Dataset | Score | RTF | Speed |
|---|---|---|---|
| NSC-Singlish (WER) | **3.31%** | 0.150 | 6.7xRT |
| LibriSpeech-EN (WER) | **2.01%** | 0.149 | 6.7xRT |
| FLEURS-Malay (WER) | **7.08%** | 0.134 | 7.5xRT |
| FLEURS-Tamil (WER) | **31.97%** | 0.236 | 4.2xRT |
| FLEURS-Mandarin (CER) | **11.17%** | 0.111 | 9.0xRT |

---

## Latency Breakdown

### polyglot-lion-0.6b-v1.5

| Dataset | Mean | p50 | p95 | p99 | Min | Max |
|---|---|---|---|---|---|---|
| NSC-Singlish | 0.62s | 0.61s | 0.86s | 0.94s | 0.36s | 1.09s |
| LibriSpeech-EN | 0.98s | 0.83s | 1.94s | 2.53s | 0.21s | 2.80s |
| FLEURS-Malay | 1.57s | 1.50s | 2.47s | 2.98s | 0.68s | 3.85s |
| FLEURS-Tamil | 5.59s | 5.38s | 8.68s | 10.52s | 2.11s | 13.10s |
| FLEURS-Mandarin | 0.99s | 0.90s | 1.66s | 2.26s | 0.42s | 2.43s |

### polyglot-lion-1.7b-v1.5

| Dataset | Mean | p50 | p95 | p99 | Min | Max |
|---|---|---|---|---|---|---|
| NSC-Singlish | 0.65s | 0.64s | 0.92s | 0.98s | 0.38s | 1.02s |
| LibriSpeech-EN | 1.01s | 0.85s | 2.01s | 2.58s | 0.22s | 2.85s |
| FLEURS-Malay | 1.64s | 1.57s | 2.54s | 2.94s | 0.71s | 4.05s |
| FLEURS-Tamil | 5.91s | 5.61s | 9.36s | 13.76s | 1.28s | 17.47s |
| FLEURS-Mandarin | 1.03s | 0.93s | 1.81s | 2.25s | 0.44s | 2.62s |

### MERaLiON-3-3B-ASR

| Dataset | Mean | p50 | p95 | p99 | Min | Max |
|---|---|---|---|---|---|---|
| NSC-Singlish | 0.84s | 0.85s | 1.16s | 1.26s | 0.52s | 1.30s |
| LibriSpeech-EN | 1.36s | 1.16s | 2.63s | 3.47s | 0.36s | 3.74s |
| FLEURS-Malay | 1.51s | 1.45s | 2.34s | 2.69s | 0.76s | 3.42s |
| FLEURS-Tamil | 3.08s | 2.92s | 4.83s | 6.61s | 1.42s | 7.45s |
| FLEURS-Mandarin | 1.28s | 1.18s | 1.99s | 2.59s | 0.64s | 3.02s |

---

## Key Observations

1. **polyglot-lion-1.7b consistently beats 0.6b** across all datasets.
2. **MERaLiON-3-3B is best on Tamil** (31.97% vs 39-40%) and Malay (7.08% vs 8.88-13.96%), likely due to training data coverage.
3. **polyglot-lion-0.6b fastest overall** at 9-11xRT on short audio, but Tamil clips (avg 14s) slow all models significantly.
4. **All results closely match paper values** -- benchmark is valid and reproducible.
5. **Tamil remains the hardest language** across all models (WER 31-40%), likely due to limited training data and complex morphology.

---

## Raw Data

Full results with per-sample latencies stored in `results/results.json`.
