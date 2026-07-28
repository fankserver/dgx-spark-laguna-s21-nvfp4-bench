# TP=2 RC1 concurrent latency probe — 2026-07-28

This is a separate experiment from the historic TP=1/vLLM 0.25.1 single-stream sweep. It evaluates the live dual-DGX-Spark RC1 service under concurrent long prefills.

## Configuration and method

- **Target:** Laguna-S-2.1-NVFP4 RC1 (`2a38deef…`), vLLM 0.26.0, TP=2 across the two GB10s.
- **Common args:** 1,048,576 max context; `--max-num-seqs 3`; `--max-num-batched-tokens 16384`; `--max-num-scheduled-tokens 4096`; explicit `--kv-cache-memory-bytes 60885130156`; prefix caching; async scheduling; Marlin MoE.
- **Probe:** `scripts/tp2_concurrent_latency_bench.py`, streaming requests with unique ~98,312-token prompts. Per-request salts occur in every block so prefix caching cannot make the cold prefill a cache hit. It runs c=1 then c=3 after a short 8k warmup.
- **Limits:** one sample per concurrency/variant, `temperature=0.7`, and unconstrained natural completion length (8–114 tokens). Thus TTFT is the sound prefill comparison; raw E2E is not a controlled decode comparison.

## Raw results

| variant | scheduled token budget | c=1 TTFT / E2E | c=3 median TTFT / E2E | c=3 completion tokens | raw output |
|---|---:|---:|---:|---:|---|
| DFlash n=7 | 16,381 effective | 67.35 / 67.80 s | 146.43 / 216.51 s | 8, 10, 87 | [`n7`](results/tp2_rc1_n7_sched16381_96k_c1c3.json) |
| DFlash n=3 | 4,096 | 66.69 / 70.39 s | 134.96 / 139.24 s | 8, 9, 114 | [`n3`](results/tp2_rc1_n3_sched4096_96k_c1c3.json) |
| no DFlash | 4,096 | **65.97 / 66.23 s** | **132.45 / 201.20 s** | 24, 66, 93 | [`n0`](results/tp2_rc1_n0_sched4096_96k_c1c3.json) |

The n=7→n=3 comparison is confounded by changing both draft depth and scheduled-token budget. The n=3↔n=0 comparison is fair on scheduler settings: no DFlash was 2.51 s (1.9%) faster at c=3 median TTFT, but one sample is not enough to call that a latency win.

## Capacity and safety result

This is decisive independently of the latency-noise caveat. At the same 60.89 GB explicit aggregate-cache cap:

| variant | target KV tokens | 1M logical concurrency | GPU process / node | idle `MemAvailable` (.79 / .80) |
|---|---:|---:|---:|---:|
| DFlash n=3 | 3,105,965 | 2.96× | ~101.35 GiB | not sampled in this exact run |
| no DFlash | **4,523,884** | **4.31×** | **94.41 GiB** | **13.25 / 15.07 GiB** |

Removing DFlash supplies **45.7% more target-KV capacity** and materially restores GB10 UMA reserve. The older n=7 measurement was at the same 2.96× KV capacity and had only roughly 6–7 GiB available during interactive load.

## Current conclusion

For the 1M/multi-session service, leave **no DFlash + 4096 scheduled tokens** active provisionally. It is at least TTFT-neutral in this matched probe, removes the drafter's large target-KV tax, and is safer on UMA. DFlash n=3 remains the likely winner for short, predictable coding completions; it needs a separate controlled decode benchmark (fixed completion length / deterministic sampling) before reintroducing it to this long-context lane.
