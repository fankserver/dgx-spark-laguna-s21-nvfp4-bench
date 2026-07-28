#!/usr/bin/env python3
"""Aggregate llama-benchy per-config JSON results into CSV + combined JSON + a
comparison markdown with graphs. Stdlib + matplotlib only.

Usage: aggregate_and_plot.py <results_dir> <out_dir>
  results_dir: contains <label>.json (llama-benchy) + optional prefix_probe.txt
"""
import sys, os, json, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# label -> human-readable config knobs (for tables)
CONFIG_META = {
    "A_nvfp4_n0":        dict(kv="auto", spec="none (n=0)",   extra="-",             maxlen=262144),
    "D_dflash_n1":       dict(kv="auto", spec="DFlash n=1",   extra="-",             maxlen=262144),
    "C_dflash_n3":       dict(kv="auto", spec="DFlash n=3",   extra="-",             maxlen=262144),
    "B_dflash_n7":       dict(kv="auto", spec="DFlash n=7",   extra="-",             maxlen=262144),
    "E_dflash_n15":      dict(kv="auto", spec="DFlash n=15",  extra="-",             maxlen=262144),
    "G_dflash_n7_fp8kv": dict(kv="fp8",  spec="DFlash n=7",   extra="-",             maxlen=262144),
    "F_nvfp4_n0_eager":  dict(kv="auto", spec="none (n=0)",   extra="enforce-eager", maxlen=262144),
}
ORDER = list(CONFIG_META.keys())

def load(results_dir):
    data = {}
    TP2_LABELS = {"tp2_rc1_n7_sched16381_96k_c1c3", "tp2_rc1_n3_sched4096_96k_c1c3",
                  "tp2_rc1_n0_sched4096_96k_c1c3", "warmup_tp2_rc1_n0_sched4096_8k_c1c3",
                  "warmup_tp2_rc1_n3_sched4096_8k_c1c3", "smoke_n7_current_2k_c1"}
    REJECT_SUFFIXES = (".stdout", ".pc", "_warmup_", "_smoke_", "tp2_rc1_")
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        label = os.path.splitext(os.path.basename(f))[0]
        if label.endswith(".stdout") or label.endswith(".pc") or label == "prefix_probe":
            continue  # .pc = prefix-caching pass, handled separately
        if label in TP2_LABELS or any(s in label for s in REJECT_SUFFIXES):
            continue  # TP=2 concurrent results use a different format
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"skip {f}: {e}"); continue
        rows = []
        def safe_mean(d):
            if d is None: return float("nan")
            v = d.get("mean")
            return round(v, 2) if v is not None else float("nan")
        for b in d.get("benchmarks", []):
            rows.append(dict(
                context=b.get("context_size"),
                prompt=b.get("prompt_size"),
                gen=b.get("response_size"),
                prefill_tps=safe_mean(b.get("pp_throughput")),
                decode_tps=safe_mean(b.get("tg_throughput")),
                peak_tps=safe_mean(b.get("peak_throughput")),
                ttft_ms=round((b.get("e2e_ttft") or {}).get("mean", float("nan")), 1),
            ))
        rows.sort(key=lambda r: (r["context"] if r["context"] is not None else 0))
        data[label] = dict(meta=CONFIG_META.get(label, {}), pc_enabled=d.get("prefix_caching_enabled"),
                           version=d.get("version"), rows=rows)
    return data

def write_csv(data, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "kv", "spec", "extra", "max_model_len", "context_tokens",
                    "prefill_tps", "decode_tps", "peak_tps", "ttft_ms"])
        for label in [l for l in ORDER if l in data] + [l for l in data if l not in ORDER]:
            m = data[label]["meta"]
            for r in data[label]["rows"]:
                w.writerow([label, m.get("kv"), m.get("spec"), m.get("extra"), m.get("maxlen"),
                            r["context"], r["prefill_tps"], r["decode_tps"], r["peak_tps"], r["ttft_ms"]])

def line_plot(data, metric, ylabel, title, path, logy=False):
    plt.figure(figsize=(9, 5.5))
    for label in [l for l in ORDER if l in data]:
        rows = data[label]["rows"]
        xs = [r["context"] for r in rows]
        ys = [r[metric] for r in rows]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("context depth (tokens)"); plt.ylabel(ylabel); plt.title(title)
    if logy: plt.yscale("log")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def grouped_bar(data, contexts, metric, ylabel, title, path):
    labels = [l for l in ORDER if l in data]
    if not labels: return
    n = len(contexts); width = 0.8 / n
    import numpy as np
    x = np.arange(len(labels))
    plt.figure(figsize=(max(10, len(labels) * 1.5), 5.5))
    for j, ctx in enumerate(contexts):
        vals = []
        for l in labels:
            row = next((r for r in data[l]["rows"] if r["context"] == ctx), None)
            vals.append(row[metric] if (row and row[metric] == row[metric]) else 0)
        pos = x + (j - (n - 1) / 2) * width
        bars = plt.bar(pos, vals, width, label=f"ctx={ctx}")
        for p, v in zip(pos, vals):
            if v: plt.text(p, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
    plt.xticks(x, labels, rotation=30, ha="right", fontsize=8)
    plt.ylabel(ylabel); plt.title(title); plt.legend(fontsize=8, title="context depth")
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def analyze_prefix_caching(results_dir, data, gdir):
    """For each <label>.pc.json, compare cold TTFT vs prefix-cached TTFT per depth."""
    out = {}
    for label in list(data.keys()):
        pcf = os.path.join(results_dir, f"{label}.pc.json")
        if not os.path.exists(pcf):
            continue
        try:
            pc = json.load(open(pcf))
        except Exception:
            continue
        cold = {r["context"]: r for r in data[label]["rows"]}
        rows = []
        for b in pc.get("benchmarks", []):
            if b.get("is_context_prefill_phase"):
                continue  # skip the context-load step; we want cached inference
            ctx = b.get("context_size")
            c = cold.get(ctx)
            if not c:
                continue
            cached_ttft = (b.get("e2e_ttft") or {}).get("mean")
            if cached_ttft is None:
                continue  # depth truncated by a node crash mid-pass
            cold_ttft = c["ttft_ms"]
            rows.append(dict(context=ctx, cold_ttft=round(cold_ttft, 0),
                             cached_ttft=round(cached_ttft, 0),
                             speedup=round(cold_ttft / cached_ttft, 2) if cached_ttft else None))
        if rows:
            out[label] = rows
    if not out:
        return []
    # graph: cold vs cached TTFT
    plt.figure(figsize=(9, 5.5))
    for label, rows in out.items():
        xs = [r["context"] for r in rows]
        plt.plot(xs, [r["cold_ttft"] for r in rows], marker="o", linestyle="--", label=f"{label} cold")
        plt.plot(xs, [r["cached_ttft"] for r in rows], marker="s", label=f"{label} cached")
    plt.xlabel("context depth (tokens)"); plt.ylabel("TTFT (ms)")
    plt.title("TTFT: cold vs prefix-cached (reused context)"); plt.yscale("log")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(gdir, "prefix_cache_ttft.png"), dpi=120); plt.close()
    # markdown
    md = ["## Prefix caching — cold vs cached TTFT (the multi-turn/agent win)\n",
          "`--enable-prefix-caching` two-step measurement: reuse a cached context vs re-prefill it. "
          "This is what a multi-turn agent sees when its prefix is stable.\n"]
    for label, rows in out.items():
        md.append(f"\n**{label}**\n")
        md.append("| context | cold TTFT (ms) | cached TTFT (ms) | speedup |")
        md.append("|--:|--:|--:|--:|")
        for r in rows:
            md.append(f"| {r['context']} | {r['cold_ttft']:.0f} | {r['cached_ttft']:.0f} | {r['speedup']}x |")
    md.append("\n![prefix_cache_ttft](graphs/prefix_cache_ttft.png)\n")
    return md

def read_prefix_probe(results_dir):
    p = os.path.join(results_dir, "prefix_probe.txt")
    if not os.path.exists(p): return None
    txt = open(p).read()
    for line in txt.splitlines():
        if line.startswith("JSON "):
            try: return json.loads(line[5:])
            except Exception: return None
    return txt

TP2_CONCURRENT_LABELS = {
    "tp2_rc1_n7_sched16381_96k_c1c3": "DFlash n=7",
    "tp2_rc1_n3_sched4096_96k_c1c3": "DFlash n=3",
    "tp2_rc1_n0_sched4096_96k_c1c3": "no DFlash",
}

def load_tp2_concurrent(results_dir):
    data = {}
    for label, display in TP2_CONCURRENT_LABELS.items():
        path = os.path.join(results_dir, f"{label}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        items = []
        for b in d.get("batches", []):
            for r in b.get("requests", []):
                u = r.get("usage", {}) or {}
                items.append(dict(
                    concurrency=b["concurrency"],
                    ttft_s=r["ttft_s"],
                    e2e_s=r["e2e_s"],
                    completion_tokens=u.get("completion_tokens", 0),
                    error=r["error"],
                ))
        if items:
            data[display] = items
    return data

def plot_tp2_concurrent_bars(data, gdir):
    variants = [k for k in TP2_CONCURRENT_LABELS.values() if k in data]
    if not variants:
        return
    concurrency_levels = sorted(set(x["concurrency"] for v in data.values() for x in v))
    n_conc = len(concurrency_levels)
    n_var = len(variants)
    x = np.arange(n_conc)
    width = 0.7 / n_var
    for metric, ylabel, title, fn in [
        ("ttft_s", "median TTFT (s)", "TP=2 concurrent 96k prefill — TTFT", "tp2_ttft_concurrent.png"),
        ("e2e_s", "median E2E (s)", "TP=2 concurrent 96k prefill — E2E", "tp2_e2e_concurrent.png"),
    ]:
        plt.figure(figsize=(max(6, n_conc * 2.5), 5.5))
        for j, var in enumerate(variants):
            items = data[var]
            vals = []
            for c in concurrency_levels:
                matched = [x[metric] for x in items if x["concurrency"] == c and x[metric] is not None]
                vals.append(np.median(matched) if matched else 0)
            pos = x + (j - (n_var - 1) / 2) * width
            bars = plt.bar(pos, vals, width, label=var)
            for p, v in zip(pos, vals):
                if v:
                    plt.text(p, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        plt.xticks(x, [f"c={c}" for c in concurrency_levels], fontsize=10)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(fontsize=9)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(gdir, fn), dpi=120)
        plt.close()
        print(f"wrote {fn}")

def main():
    results_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    gdir = os.path.join(out_dir, "graphs"); os.makedirs(gdir, exist_ok=True)
    data = load(results_dir)
    if not data:
        print("no data found"); return
    json.dump(data, open(os.path.join(out_dir, "dataset.json"), "w"), indent=2)
    write_csv(data, os.path.join(out_dir, "dataset.csv"))
    line_plot(data, "decode_tps", "decode tok/s", "Decode throughput vs context depth",
              os.path.join(gdir, "decode_vs_context.png"))
    line_plot(data, "ttft_ms", "TTFT (ms)", "Time-to-first-token vs context depth (pp=512)",
              os.path.join(gdir, "ttft_vs_context.png"), logy=True)
    line_plot(data, "prefill_tps", "prefill tok/s", "Prefill throughput vs context depth",
              os.path.join(gdir, "prefill_vs_context.png"))
    # representative contexts present in the data for grouped bars
    all_ctx = sorted({r["context"] for d in data.values() for r in d["rows"]})
    pick = [c for c in [0, 8192, 16384, 32768] if c in all_ctx] or all_ctx[::max(1, len(all_ctx)//4)]
    grouped_bar(data, pick, "decode_tps", "decode tok/s",
                "Decode tok/s by config across context depths", os.path.join(gdir, "decode_grouped.png"))
    grouped_bar(data, pick, "ttft_ms", "TTFT (ms)",
                "TTFT by config across context depths", os.path.join(gdir, "ttft_grouped.png"))
    pc_md = analyze_prefix_caching(results_dir, data, gdir)
    prefix = read_prefix_probe(results_dir)

    # markdown
    md = ["# Laguna-S-2.1-NVFP4 on 1× DGX Spark (GB10) — performance sweep",
          "",
          "Standard benchmark: **llama-benchy** (llama-bench-style). Model kept constant "
          "(poolside/Laguna-S-2.1-NVFP4, TP=1). Each config redeployed fresh; "
          "context-depth sweep with pp=512, tg=256, runs=2, latency-mode=generation.",
          ""]
    v = next(iter(data.values())).get("version")
    md.append(f"_llama-benchy {v} · single-stream (concurrency=1)_\n")
    md.append("## Configs\n")
    md.append("| config | KV dtype | speculation | extra | max_model_len |")
    md.append("|---|---|---|---|---|")
    for l in [x for x in ORDER if x in data]:
        m = data[l]["meta"]
        md.append(f"| `{l}` | {m.get('kv')} | {m.get('spec')} | {m.get('extra')} | {m.get('maxlen')} |")
    md.append("")
    # summary table at key contexts
    md.append("## Summary — decode tok/s and TTFT by context\n")
    md.append("| config | dec@0 | dec@4k | dec@16k | dec@32k | ttft@16k(ms) | ttft@32k(ms) |")
    md.append("|---|--:|--:|--:|--:|--:|--:|")
    def cell(label, ctx, metric):
        r = next((x for x in data[label]["rows"] if x["context"] == ctx), None)
        return f"{r[metric]:.1f}" if r and r[metric] == r[metric] else "—"
    for l in [x for x in ORDER if x in data]:
        md.append(f"| `{l}` | {cell(l,0,'decode_tps')} | {cell(l,4096,'decode_tps')} | "
                  f"{cell(l,16384,'decode_tps')} | {cell(l,32768,'decode_tps')} | "
                  f"{cell(l,16384,'ttft_ms')} | {cell(l,32768,'ttft_ms')} |")
    md.append("")
    md.append("## Full data\n\nSee `dataset.csv` / `dataset.json`. Per-config, per-context rows:\n")
    md.append("| config | ctx | prefill t/s | decode t/s | peak t/s | TTFT ms |")
    md.append("|---|--:|--:|--:|--:|--:|")
    for l in [x for x in ORDER if x in data]:
        for r in data[l]["rows"]:
            md.append(f"| `{l}` | {r['context']} | {r['prefill_tps']} | {r['decode_tps']} | {r['peak_tps']} | {r['ttft_ms']} |")
    md.append("")
    md += pc_md
    if prefix:
        md.append("## Prefix-cache effectiveness probe (config B, fp8)\n")
        if isinstance(prefix, dict) and prefix.get("prefix_reuse"):
            md.append("| shared prefix tok | TTFT cold (s) | TTFT warm (s) | speedup | warm hits/queries |")
            md.append("|--:|--:|--:|--:|--:|")
            for r in prefix["prefix_reuse"]:
                md.append(f"| {r['prefix_tok']} | {r['ttft_cold']:.2f} | {r['ttft_warm']:.2f} | "
                          f"{r['speedup']:.2f}x | {r['warm_hits']}/{r['warm_queries']} |")
        else:
            md.append("```\n" + str(prefix)[:1500] + "\n```")
        md.append("")
    # ── TP=2 concurrent latency graphs ──────────────────────────────
    tp2_data = load_tp2_concurrent(results_dir)
    if tp2_data:
        plot_tp2_concurrent_bars(tp2_data, gdir)
        md.append("## TP=2 concurrent 96k probe\n")
        md.append("![TTFT concurrent](graphs/tp2_ttft_concurrent.png)\n")
        md.append("![E2E concurrent](graphs/tp2_e2e_concurrent.png)\n")
        md.append("See [tp2-rc1-concurrent-2026-07-28.md](tp2-rc1-concurrent-2026-07-28.md) for analysis.\n")
    md.append("## Graphs\n")
    for g in ["decode_vs_context.png", "ttft_vs_context.png", "prefill_vs_context.png",
              "decode_grouped.png", "ttft_grouped.png"]:
        if os.path.exists(os.path.join(gdir, g)):
            md.append(f"![{g}](graphs/{g})\n")
    open(os.path.join(out_dir, "comparison.md"), "w").write("\n".join(md))
    print(f"wrote dataset.csv, dataset.json, comparison.md, graphs/ to {out_dir}")
    print(f"configs: {list(data.keys())}")

if __name__ == "__main__":
    main()
