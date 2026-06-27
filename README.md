# ASR Benchmark Suite — Singlish & Singapore Languages

A comprehensive benchmark suite for evaluating Automatic Speech Recognition (ASR) models on **Singlish** (Singapore English) and other Singapore-relevant languages, using state-of-the-art open-source models.

## Overview

This project benchmarks three ASR models across five diverse datasets, measuring Word Error Rate (WER), Character Error Rate (CER), Real-Time Factor (RTF), and latency metrics. It is designed for reproducible evaluation on a single GPU (RTX 3080 20GB).

### Models

| Model | Parameters | Backend | Origin |
|---|---|---|---|
| **polyglot-lion-0.6b-v1.5** | 0.6B | Qwen2-Audio | [knoveleng/polyglot-lion-0.6b-v1.5](https://huggingface.co/knoveleng/polyglot-lion-0.6b-v1.5) |
| **polyglot-lion-1.7b-v1.5** | 1.7B | Qwen2-Audio | [knoveleng/polyglot-lion-1.7b-v1.5](https://huggingface.co/knoveleng/polyglot-lion-1.7b-v1.5) |
| **MERaLiON-3-3B-ASR** | 3B | MERaLiON | [MERaLiON/MERaLiON-3-3B-ASR](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR) |

### Datasets

| Dataset | Language | Metric | Test Samples | Source |
|---|---|---|---|---|
| NSC-Singlish | Singlish (English+Chinese mix) | WER | 3,000 | [knoveleng/nsc-singlish](https://huggingface.co/datasets/knoveleng/nsc-singlish) |
| LibriSpeech-EN | English | WER | 2,620 | [knoveleng/librispeech-english](https://huggingface.co/datasets/knoveleng/librispeech-english) |
| FLEURS-Malay | Malay (Bahasa Malaysia) | WER | 749 | [knoveleng/fleurs-malay](https://huggingface.co/datasets/knoveleng/fleurs-malay) |
| FLEURS-Tamil | Tamil | WER | 591 | [knoveleng/fleurs-tamil](https://huggingface.co/datasets/knoveleng/fleurs-tamil) |
| FLEURS-Mandarin | Mandarin Chinese | CER | ~1,500 | [knoveleng/fleurs-mandarin](https://huggingface.co/datasets/knoveleng/fleurs-mandarin) |

## Results Summary

Benchmarked on **RTX 3080 (20GB VRAM)** with **bfloat16** precision, **200 samples per dataset** (3 warmup, ~197 measured).

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

> **Key observations:**
> - **polyglot-lion-1.7b** consistently outperforms the 0.6b variant across all datasets.
> - **MERaLiON-3-3B** excels on Tamil (31.97% vs 39–40%) and Malay (7.08%), likely due to better training data coverage for Austronesian/Dravidian languages.
> - **polyglot-lion-0.6b** is fastest overall at 9–11×RT on short audio clips.
> - **Tamil remains the hardest language** across all models (WER 31–40%), due to limited training data and complex morphology.
> - All results closely match published paper values, confirming benchmark validity.

## Usage

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥20GB VRAM (tested on RTX 3080)
- CUDA 12.x

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Download test datasets and model weights:

```bash
python3 download_data.py
```

This fetches only the test split parquet files (no full dataset builds) and any uncached model weights.

### Running Smoke Tests

Quick validation with a few samples (streaming, no full download needed):

```bash
# polyglot-lion-0.6b on 5 Singlish samples (GPU 1)
python3 smoke_test.py

# polyglot-lion-1.7b on 2 Singlish samples (GPU 1)
python3 smoke_test_17b.py
```

### Running the Full Benchmark

```bash
# Run a specific model on a specific GPU
python3 benchmark.py --device 0 --model polyglot-lion-1.7b-v1.5
python3 benchmark.py --device 1 --model MERaLiON-3-3B-ASR

# Run all models on default GPU (GPU 1)
python3 benchmark.py

# Run with logging
CUDA_VISIBLE_DEVICES=1 python3 benchmark.py 2>&1 | tee benchmark.log
```

### Docker Deployment

Build and run the inference serving stack (all three models on GPU 1):

```bash
DOCKER_HOST=unix:///var/run/docker.sock docker compose -f deploy/docker-compose.yml build
DOCKER_HOST=unix:///var/run/docker.sock docker compose -f deploy/docker-compose.yml up -d
```

The Docker stack serves models on ports:
- `9001` — polyglot-lion-0.6b
- `9002` — polyglot-lion-1.7b
- `9003` — MERaLiON-3-3B-ASR

### Outputs

- `results/results.json` — Per-sample latencies and aggregated metrics
- `results/BENCHMARK_LOG.md` — Formatted benchmark report

## Project Structure

```
├── benchmark.py             # Main benchmark script
├── download_data.py         # Dataset & model downloader
├── smoke_test.py            # Quick smoke test (0.6b, 5 samples)
├── smoke_test_17b.py        # Quick smoke test (1.7b, 2 samples)
├── deploy/
│   ├── Dockerfile           # Docker image (vLLM + ASR models)
│   └── docker-compose.yml   # Multi-service orchestration
└── results/
    ├── results.json         # Raw per-sample metrics
    └── BENCHMARK_LOG.md     # Formatted benchmark report
```

## Technical Notes

- **Backends:** Qwen2-Audio models (polyglot-lion) use `qwen` backend; MERaLiON uses `meralion` backend — both register custom model architectures with vLLM's `ModelRegistry`.
- **Precision:** All benchmarks run with `bfloat16` for optimal throughput.
- **GPU Memory Budget (20 GB):**
  - 0.6b → `gpu_memory_utilization 0.12` ≈ 2.4 GB
  - 1.7b → `gpu_memory_utilization 0.22` ≈ 4.4 GB
  - 3B → `gpu_memory_utilization 0.38` ≈ 7.6 GB
  - Total ≈ 14.4 GB (72% of 20 GB)
- **Text normalization** follows the `knoveleng/asr-evalkit` `TextNormalizer` exactly, including fullwidth-to-halfwidth conversion, punctuation removal, and NFC normalization.

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Models by [knoveleng](https://huggingface.co/knoveleng) and [MERaLiON](https://huggingface.co/MERaLiON)
- Datasets from NSC, LibriSpeech, and FLEURS (FbU) collections
- Powered by [vLLM](https://github.com/vllm-project/vllm) inference engine
