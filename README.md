# 1× DGX Spark (GB10) — Laguna-S-2.1 NVFP4 performance benchmarks

Single-stream inference benchmarks for **poolside Laguna-S-2.1** (118B-A8B coding MoE)
quantized to **NVFP4**, served with **vLLM 0.25.1 on a single NVIDIA GB10 / DGX Spark
(TP=1)**. Measured with the standard **[llama-benchy](https://github.com/eugr/llama-benchy)**
tool, using the **same context-depth set as the
[Qwen3.5-122B bench](https://github.com/fankserver/dgx-spark-qwen3.5-122b-bench)** so the
numbers are directly comparable. Sweeps 7 engine configs × 14 depths (98 datapoints), plus a
TP=1-vs-TP=2 scaling spot-check.

**Headline:** Laguna-S-2.1-NVFP4 is an excellent **single-Spark** coder — it fits in 67 GiB
on one GB10 with a **1.37M-token KV pool** (5.24× concurrency at 262K context), and unlike
GLM-4.7, **speculative decoding actually works here**: the shipped DFlash drafter delivers a
real speedup, best at **`num_speculative_tokens=3`** (25.5 tok/s vs 18.3 baseline on generic
text; **~2.35× on actual coding prompts**). Decode is remarkably **flat across context**
(18.3→16.5 tok/s over 0→65k) thanks to the sliding-window attention. TP=2 across two nodes
*does* scale (~1.7× decode, ~2.0× prefill) but is **unstable** on this hardware — so **TP=1 is
the recommended config**.

## Environment

| | |
|---|---|
| **Model** | `poolside/Laguna-S-2.1-NVFP4` — Laguna-S-2.1 118B-A8B MoE (`model_type: laguna`), 48 layers (12 full-attention + 36 sliding-window@512), 256 experts (top-10) + 1 shared, GQA 8 KV heads, per-head softplus gating, YaRN to 1,048,576 ctx. 67.0 GiB weights |
| **Quantization** | **NVFP4** (weights); DFlash drafter `poolside/Laguna-S-2.1-DFlash-NVFP4` (NVFP4, 2.1 GiB) |
| **Serving engine** | vLLM **0.25.1** (`vllm-v0251-clamp`, SM121 aarch64 build) — resolves `LagunaForCausalLM` + `DFlashLagunaForCausalLM` natively |
| **Topology** | **TP=1, single GB10** (no fabric). `--tool-call-parser poolside_v1 --reasoning-parser poolside_v1` |
| **GPU** | NVIDIA **GB10** (DGX Spark), SM 12.1, 128 GB / 121.6 GiB unified LPDDR5X, ~273 GB/s |
| **Driver / CUDA** | 580.159.03 / CUDA 13.0 · Ubuntu 24.04, kernel 6.17.0-1026-nvidia, aarch64 |
| **Serving params** | `--gpu-memory-utilization 0.85 --max-model-len 262144 --max-num-seqs 32`, prefix caching on |
| **Benchmark params** | `--pp 512 --tg 256 --exact-tg --runs 2 --latency-mode generation`, depths `0 512 1024 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536` (Qwen-122B set) |
| **Date** | 2026-07-22 |

## Configs tested

Same model throughout; each config changes **one knob** from the `A` baseline. `n` =
`num_speculative_tokens` (DFlash draft depth); `n0` = speculation off.

| config | spec | KV | other | isolates |
|---|---|---|---|---|
| `A_nvfp4_n0` | none | auto | — | baseline |
| `D_dflash_n1` | DFlash n=1 | auto | — | draft depth |
| `C_dflash_n3` | DFlash n=3 | auto | — | **draft depth (best)** |
| `B_dflash_n7` | DFlash n=7 | auto | — | draft depth |
| `E_dflash_n15` | DFlash n=15 | auto | — | draft depth (poolside default) |
| `G_dflash_n7_fp8kv` | DFlash n=7 | **fp8** | — | KV dtype on the spec path |
| `F_nvfp4_n0_eager` | none | auto | `--enforce-eager` | value of CUDA graphs |

## Contents

- [`findings.md`](findings.md) — verdict, DFlash tuning curve, TP scaling, the vLLM-version story
- [`comparison.md`](comparison.md) — auto-generated per-datapoint tables + graphs
- [`dataset.csv`](dataset.csv) / [`dataset.json`](dataset.json) — every datapoint (7×14 = 98)
- `graphs/` — decode/TTFT/prefill vs context, grouped bars
- `results/` — raw `llama-benchy` JSON per config, `*.spec.txt` (spec-accept metrics), `TP2_nvfp4_n0.json` (scaling spot-check)
- `scripts/` — serve launcher, bench driver, aggregator

## Reproduce

```bash
hf download poolside/Laguna-S-2.1-NVFP4              # 67 GiB, one node
hf download poolside/Laguna-S-2.1-DFlash-NVFP4       # 2.1 GiB drafter
# serve (TP=1) — see scripts/laguna-serve2.sh; needs a vLLM 0.25.0+ image with laguna support
python3 -m venv benchy-venv && ./benchy-venv/bin/pip install -U llama-benchy
bash scripts/laguna_bench_driver.sh                 # redeploys per config
python3 scripts/aggregate_and_plot.py results .
```

## Caveats

Single-stream (concurrency=1); `runs=2`. Spec acceptance is workload-dependent — benchy's
book text understates the coding-workload gain (see findings). fp8 KV runs without calibrated
scales. Numbers are specific to this exact engine/model/hardware/date.
