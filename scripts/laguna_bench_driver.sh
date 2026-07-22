#!/usr/bin/env bash
# Laguna-S-2.1-NVFP4 solo TP=1 bench — Qwen-122B depth set for cross-comparison.
# Redeploys per config on .79:8080 (api-key → alloy scrapes). A + B (n7) run separately.
set -uo pipefail
KEY=$(cat /home/factory/hy3-deploy/api-key)
OUT=/home/factory/laguna-bench-results
BENCHY=/home/factory/benchy-venv/bin/llama-benchy
DEPTHS="0 512 1024 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536"
mkdir -p "$OUT"
DF='{"model":"poolside/Laguna-S-2.1-DFlash-NVFP4","num_speculative_tokens":%d,"method":"dflash"}'

wait_healthy(){ for i in $(seq 1 90); do
  curl -sf -m3 http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0
  docker logs vllm_laguna 2>&1 | grep -qE "EngineCore failed|ValueError|Traceback" && return 1
  sleep 15; done; return 1; }
warmup(){ curl -sf -m180 http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d '{"model":"spark","messages":[{"role":"user","content":"hi"}],"max_tokens":32}' >/dev/null 2>&1; }
bench(){ HF_HUB_DISABLE_IMPLICIT_TOKEN=1 "$BENCHY" --base-url http://127.0.0.1:8080/v1 --api-key "$KEY" \
  --model poolside/Laguna-S-2.1-NVFP4 --skip-coherence --depth $DEPTHS --pp 512 --tg 256 --exact-tg \
  --runs 2 --latency-mode generation --format json --save-result "$OUT/$1.json" > "$OUT/$1.stdout.txt" 2>&1; }

run_cfg(){ # label extra-serve-args...
  local label="$1"; shift
  [ -s "$OUT/$label.json" ] && { echo "SKIP $label"; return; }
  echo "=== $(date +%H:%M:%S) DEPLOY $label"
  /home/factory/laguna-serve2.sh 0.85 262144 "$@" >/dev/null 2>&1
  if wait_healthy; then
    warmup; echo "$(date +%H:%M:%S) bench $label"; bench "$label"
    # snapshot spec metrics if any
    curl -s http://127.0.0.1:8080/metrics 2>/dev/null | grep -E "spec_decode_num_(draft_tokens|accepted_tokens)_total" > "$OUT/$label.spec.txt" 2>/dev/null || true
    echo "$(date +%H:%M:%S) done $label"
  else echo "$(date +%H:%M:%S) DEPLOY FAILED $label"; docker logs --tail 80 vllm_laguna > "$OUT/$label.deployfail.txt" 2>&1 || true; fi
}

run_cfg C_dflash_n3  --speculative-config "$(printf "$DF" 3)"
run_cfg D_dflash_n1  --speculative-config "$(printf "$DF" 1)"
run_cfg E_dflash_n15 --speculative-config "$(printf "$DF" 15)"
run_cfg F_nvfp4_n0_eager --enforce-eager
run_cfg G_dflash_n7_fp8kv --kv-cache-dtype fp8 --speculative-config "$(printf "$DF" 7)"
echo "=== ALL DONE $(date +%H:%M:%S)"
