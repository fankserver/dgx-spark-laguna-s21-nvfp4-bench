#!/bin/bash
# Laguna-S-2.1-NVFP4 solo TP=1 on .79. Args: [gmu] [maxlen] [extra vllm args...]
set -uo pipefail
KEY=$(cat ~/hy3-deploy/api-key)
GMU="${1:-0.85}"; MAXLEN="${2:-262144}"; shift 2 || true
docker rm -f vllm_laguna 2>/dev/null; sleep 2
docker run -d --rm --name vllm_laguna --gpus all --network host --ipc host \
  --ulimit memlock=-1:-1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_HOST_IP=192.168.178.79 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  vllm-v0251-clamp:latest \
  poolside/Laguna-S-2.1-NVFP4 \
  --served-model-name poolside/Laguna-S-2.1-NVFP4 spark \
  --tensor-parallel-size 1 \
  --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --enable-auto-tool-choice \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" --max-num-seqs 32 \
  --host 0.0.0.0 --port 8080 --api-key "$KEY" "$@"
echo "launched (gmu=$GMU maxlen=$MAXLEN extra: $*)"
