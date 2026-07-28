#!/usr/bin/env python3
"""Generate TTFT and E2E bar charts from the TP=2 concurrent latency results."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS_RENAME = {
    "tp2_rc1_n7_sched16381_96k_c1c3": "DFlash n=7",
    "tp2_rc1_n3_sched4096_96k_c1c3": "DFlash n=3",
    "tp2_rc1_n0_sched4096_96k_c1c3": "no DFlash",
}
RESULTS_DIR = os.path.join(os.path.dirname(__file__) or ".", "..", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__) or ".", "..", "graphs")
os.makedirs(OUT_DIR, exist_ok=True)

def load():
    data = {}
    for label, display in LABELS_RENAME.items():
        path = os.path.join(RESULTS_DIR, f"{label}.json")
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
        data[display] = items
    return data

def grouped_bars(data, metric, ylabel, title, filename):
    variants = [k for k in LABELS_RENAME.values() if k in data]
    if not variants:
        return
    concurrency_levels = sorted(set(x["concurrency"] for v in data.values() for x in v))
    n_conc = len(concurrency_levels)
    n_var = len(variants)
    x = np.arange(n_conc)
    width = 0.7 / n_var
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
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=120)
    plt.close()
    print(f"wrote {filename}")

def main():
    data = load()
    if not data:
        print("no tp2 concurrent results found in", RESULTS_DIR)
        sys.exit(1)
    grouped_bars(data, "ttft_s", "median TTFT (s)",
                 "TP=2 concurrent 96k prefill — TTFT", "tp2_ttft_concurrent.png")
    grouped_bars(data, "e2e_s", "median E2E (s)",
                 "TP=2 concurrent 96k prefill — E2E", "tp2_e2e_concurrent.png")

if __name__ == "__main__":
    main()
