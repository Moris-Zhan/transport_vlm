#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/moris/Project/transport_vlm_refactor
MODEL=/mnt/moris/Project/transport_vlm_refactor/Output/Mistral-Small-3.2-24B-Instruct-2506-lora-run1/merged-18800
VLLM_PORT=8000
FE_PORT=8090
# External URL prefix added by the ingress/proxy (no trailing slash)
ROOT_PATH=/a4665b22-6514-463b-8dd3-28ccb78ae0b7/traffic-collision-diagram
CONDA_ENV=/mnt/moris/conda_envs/Mistral3-fa-vllm

# ---- Activate conda env (no conda install needed in the image) ----
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
echo "[start.sh] python: $(which python)"
echo "[start.sh] vllm:   $(which vllm)"

cd "$PROJECT_DIR"

# ---- Backend: vLLM ----
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$MODEL" \
  --tokenizer "$MODEL" \
  --load_format safetensors \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --host 0.0.0.0 --port "$VLLM_PORT" &
VLLM_PID=$!

# On SIGTERM (pod deletion), stop both child processes
trap 'kill -TERM ${VLLM_PID:-} ${FE_PID:-} 2>/dev/null; wait' TERM INT

# ---- Wait for vLLM to be ready ----
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[start.sh] vLLM exited before ready"; exit 1; }
  sleep 5
done
echo "[start.sh] vLLM ready on :${VLLM_PORT}"

# ---- Frontend ----
python -m uvicorn inference_server:app \
  --app-dir demo_ori \
  --host 0.0.0.0 --port "$FE_PORT" \
  --root-path "$ROOT_PATH" \
  --proxy-headers --forwarded-allow-ips '*' \
  --log-level info &
FE_PID=$!

# If either process exits, shut down the container and let k8s restart it
wait -n "$VLLM_PID" "$FE_PID"
echo "[start.sh] a process exited, shutting down"
kill -TERM "$VLLM_PID" "$FE_PID" 2>/dev/null || true
wait
exit 1
