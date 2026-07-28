#!/usr/bin/env python3
"""Non-destructive latency probe for the dual-Spark Laguna service.

It intentionally does not deploy or remove containers.  Every prompt is unique so
prefix caching cannot turn a long prefill into a misleading cache hit.  Results
contain raw per-request TTFT/E2E measurements and selected vLLM metric snapshots.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests


def api_post(base_url: str, api_key: str, path: str, payload: dict, **kwargs):
    return requests.post(
        f"{base_url.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=kwargs.pop("timeout", 1800),
        **kwargs,
    )


def token_count(base_url: str, api_key: str, model: str, prompt: str) -> int:
    response = api_post(base_url, api_key, "/tokenize", {"model": model, "prompt": prompt})
    response.raise_for_status()
    return int(response.json()["count"])


def prompt_for_blocks(run_id: str, request_id: int, blocks: int) -> str:
    # The per-request salt is deliberately included in every block. Prefix-block
    # hashes therefore differ across simultaneous requests.
    salt = f"{run_id}-{request_id:02d}"
    body = " ".join(
        f"session-{salt} record-{index:07d}: evidence changes only after independent verification."
        for index in range(blocks)
    )
    return (
        f"This is private context for {salt}. Read it but do not quote it.\n{body}\n\n"
        "In exactly one short paragraph, state that the context was processed."
    )


def make_prompt(base_url: str, api_key: str, model: str, target_tokens: int, run_id: str, request_id: int):
    # Binary-search the number of deterministic unique blocks. Exact equality is
    # unnecessary; the measured server-side prompt_tokens is stored in results.
    low, high = 1, max(2, target_tokens // 2)
    while token_count(base_url, api_key, model, prompt_for_blocks(run_id, request_id, high)) < target_tokens:
        low, high = high, high * 2
    best_prompt, best_count = "", 0
    while low <= high:
        middle = (low + high) // 2
        candidate = prompt_for_blocks(run_id, request_id, middle)
        count = token_count(base_url, api_key, model, candidate)
        if count <= target_tokens:
            best_prompt, best_count = candidate, count
            low = middle + 1
        else:
            high = middle - 1
    return best_prompt, best_count


def metric_snapshot(base_url: str):
    try:
        text = requests.get(f"{base_url.rstrip('/')}/metrics", timeout=10).text
    except requests.RequestException as exc:
        return {"error": str(exc)}
    selected = {}
    prefixes = (
        "vllm:num_requests_",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )
    for line in text.splitlines():
        if line.startswith(prefixes):
            name, value = line.rsplit(" ", 1)
            selected[name] = value
    return selected


def stream_request(base_url: str, api_key: str, model: str, prompt: str, output_tokens: int, start_gate: threading.Event):
    start_gate.wait()
    started = time.perf_counter()
    first_event = None
    completion = ""
    usage = None
    status = None
    error = None
    try:
        with api_post(
            base_url,
            api_key,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": output_tokens,
                "temperature": 0.7,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            stream=True,
            timeout=1800,
        ) as response:
            status = response.status_code
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                if first_event is None:
                    first_event = time.perf_counter()
                data = line[6:]
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    completion += choice.get("delta", {}).get("content") or ""
    except Exception as exc:  # retain failures as evidence instead of losing the batch
        error = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    return {
        "http_status": status,
        "error": error,
        "ttft_s": None if first_event is None else first_event - started,
        "e2e_s": finished - started,
        "usage": usage,
        "completion_chars": len(completion),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.178.79:8080")
    parser.add_argument("--api-key", default=os.environ.get("LAGUNA_API_KEY"))
    parser.add_argument("--model", default="spark")
    parser.add_argument("--prompt-tokens", type=int, default=98304)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrencies", default="1,3")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("set --api-key or LAGUNA_API_KEY")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "label": args.label,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "target_prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "rounds": args.rounds,
        "batches": [],
        "metrics_before": metric_snapshot(args.base_url),
    }
    for concurrency in [int(value) for value in args.concurrencies.split(",")]:
        for round_index in range(args.rounds):
            run_id = uuid.uuid4().hex[:12]
            prompts = [make_prompt(args.base_url, args.api_key, args.model, args.prompt_tokens, run_id, index)
                       for index in range(concurrency)]
            gate = threading.Event()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(stream_request, args.base_url, args.api_key, args.model, prompt,
                                           args.output_tokens, gate)
                           for prompt, _ in prompts]
                gate.set()
                requests_result = [future.result() for future in futures]
            successful = [entry for entry in requests_result if entry["error"] is None]
            results["batches"].append({
                "concurrency": concurrency,
                "round": round_index + 1,
                "generated_prompt_tokens": [count for _, count in prompts],
                "requests": requests_result,
                "summary": {
                    "successful": len(successful),
                    "median_ttft_s": statistics.median(entry["ttft_s"] for entry in successful if entry["ttft_s"] is not None) if successful else None,
                    "median_e2e_s": statistics.median(entry["e2e_s"] for entry in successful) if successful else None,
                },
            })
    results["metrics_after"] = metric_snapshot(args.base_url)
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    output = args.output_dir / f"{args.label}.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)
    for batch in results["batches"]:
        print(f"c={batch['concurrency']} round={batch['round']} {batch['summary']}")


if __name__ == "__main__":
    main()
