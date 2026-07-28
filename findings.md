# Findings — Laguna-S-2.1-NVFP4 on 1× GB10

Complete sweep: **7 configs × 14 depths = 98 datapoints** (same depth set as the
[Qwen3.5-122B bench](https://github.com/fankserver/dgx-spark-qwen3.5-122b-bench)), plus a
TP=1-vs-TP=2 scaling spot-check. Per-datapoint tables/graphs in `comparison.md`; raw JSON in
`results/`.

## Verdict

**Ship `C_dflash_n3` (DFlash, `num_speculative_tokens=3`) on a single Spark (TP=1).** It's the
fastest config at every depth, and Laguna-S-2.1-NVFP4 is a strong single-GB10 coder: 67 GiB
weights leave a **1.37M-token KV pool** (5.24× concurrency at 262K). This is the mirror image
of the [GLM-4.7 result](https://github.com/fankserver/dgx-spark-glm-4.7-awq-bench), where
speculative decoding *hurt* — here it clearly helps.

## Decode throughput by config (tok/s)

| config | spec | @0 | @8k | @32k | @64k | draft accept |
|---|---|--:|--:|--:|--:|--:|
| A_nvfp4_n0 | none | 18.3 | 18.3 | 17.8 | 16.5 | — |
| D_dflash_n1 | n=1 | 23.4 | 21.4 | 19.5 | 20.0 | **53%** |
| **C_dflash_n3** | **n=3** | **25.5** | **23.0** | **21.1** | **21.2** | 30% |
| B_dflash_n7 | n=7 | 23.5 | 20.0 | 17.7 | 16.9 | ~13% |
| E_dflash_n15 | n=15 | 20.7 | 18.2 | 13.7 | 16.0 | 6% |
| G_dflash_n7_fp8kv | n=7 | 23.3 | 20.8 | 16.3 | 22.7 | 13% |
| F_nvfp4_n0_eager | none | 18.0 | 17.7 | 16.7 | 16.0 | — |

## The results worth remembering

1. **DFlash works — and `n=3` is the sweet spot.** 25.5 tok/s vs the 18.3 baseline on
   benchy's generic book text (**1.39×**), rising to **~2.35× (42.6 tok/s) on actual coding
   prompts** measured directly — acceptance is far higher on predictable code, which is the
   model's purpose. This directly refutes the early NVIDIA-forum reports of "abysmal ~8%
   acceptance," which were an artifact of the older vLLM 0.23.1; on **0.25.1** DFlash is
   excellent.
2. **More draft tokens is worse, not better.** Acceptance decays fast with draft depth
   (n=1: 53% → n=3: 30% → n=7: 13% → n=15: 6%), and per-position acceptance collapses past
   ~token 3. **n=15 (poolside's default) is the *worst* DFlash config** — it wastes compute
   drafting tokens that are almost always rejected. n=1–3 is the productive range.
3. **Decode is remarkably flat across context** (18.3 → 16.5 tok/s over 0 → 65k, ~10%),
   versus GLM-4.7's ~33% drop. This is the **sliding-window attention** (512-token window on
   36 of 48 layers): KV reads don't grow with context, so long contexts stay fast. Prefill is
   also strong and flat (~1,700–2,500 tok/s across the range).
4. **fp8 KV helps sustain at depth** (`G` holds 22.7 tok/s at 64k vs `B`'s 16.9 on the same
   n=7 config) — though Laguna's KV is already cheap, so the effect is smaller than on a
   full-attention model.
5. **CUDA graphs barely matter** (`A` vs `F` within ~1 tok/s) — batch-1 decode is
   bandwidth-bound. `--enforce-eager` is a safe fallback.

## TP=1 vs TP=2 — does it scale?

Spot-check on the baseline (no-spec) config, `.79`+`.80` over the 200G fabric:

| | prefill @0 | decode @0 |
|---|--:|--:|
| TP=1 (1 node) | 1,726 tok/s | 18.3 tok/s |
| TP=2 (2 nodes) | **3,469 tok/s (2.0×)** | **31.1 tok/s (1.7×)** |

**It scales** — TP=2 nearly doubles prefill and gives ~1.7× decode by splitting the model
across both GPUs' bandwidth. **But TP=2 is unstable**: the run crashed after the first depth
with `NVRM: NV_ERR_NO_MEMORY` from `_memdescAllocInternal` — the **same GPU-driver-level
2-node failure we hit repeatedly benchmarking GLM-4.7 TP=2 on this hardware**, reproduced here
on a second, unrelated model. It is not host-RAM OOM (123 GiB free after teardown); it is a
driver-level instability specific to the 2× GB10 TP=2 path.

**Conclusion:** for a stable deployment, run **TP=1** — the model fits on one node with huge
KV headroom, and TP=1 sidesteps the 2-node driver fragility entirely. TP=2 is worth ~1.7–2×
if/when the underlying NVRM instability is resolved.

## Operational notes

- **vLLM 0.25.0+ is required** (poolside). The stock 0.22.1 image does not resolve the
  `laguna` arch; `vllm-v0251-clamp` (0.25.1, SM121) registers both `LagunaForCausalLM` and
  `DFlashLagunaForCausalLM` and serves cleanly on GB10.
- It's a **reasoning/interleaved-thinking** model (thinking streams inline in `content`).
  Throughput is unaffected; for tool-eval quality some users prefer `enable_thinking:false`.
- TP=1 uses the FlashInfer attention backend without incident (the GLM FLASHINFER hard-hang
  was TP rank-1-specific; single-node has no rank-1 worker).
- **TP=2 RC1 concurrent follow-up:** see [`tp2-rc1-concurrent-2026-07-28.md`](tp2-rc1-concurrent-2026-07-28.md).
  This is deliberately separate from the TP=1 verdict: for the 1M/multi-session lane, removing
  DFlash increased target-KV capacity from 2.96× to 4.31× at the same cache cap and restored UMA reserve.
