#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST=${REMOTE_HOST:-nvidia@192.168.1.230}
REMOTE_ROOT=${REMOTE_ROOT:-/home/nvidia/qwen3-tts-api}
REMOTE_RESULTS_DIR=${REMOTE_RESULTS_DIR:-$REMOTE_ROOT/benchmarks/results}
IMAGE=${IMAGE:-qwen3-tts-api-jetson:latest}
MACHINE_NAME=${MACHINE_NAME:-jetson-agx-orin}
RUN_TAG=${RUN_TAG:-cors-$(date +%Y%m%d-%H%M%S)}
ORIGIN=${ORIGIN:-https://reader.13gxg.heiyu.space}
REQUEST_METHOD=${REQUEST_METHOD:-POST}
REQUEST_HEADERS=${REQUEST_HEADERS:-content-type}
QWEN_TTS_PORT=${QWEN_TTS_PORT:-8000}
QWEN_TTS_DEFAULT_SPEAKER=${QWEN_TTS_DEFAULT_SPEAKER:-Chelsie}
QWEN_TTS_CORS_ALLOW_ORIGINS=${QWEN_TTS_CORS_ALLOW_ORIGINS:-$ORIGIN}
QWEN_TTS_CORS_ALLOW_METHODS=${QWEN_TTS_CORS_ALLOW_METHODS:-POST,OPTIONS}
QWEN_TTS_CORS_ALLOW_HEADERS=${QWEN_TTS_CORS_ALLOW_HEADERS:-content-type,authorization}
QWEN_TTS_CORS_EXPOSE_HEADERS=${QWEN_TTS_CORS_EXPOSE_HEADERS:-X-Qwen-TTS-Segment-Count,X-Qwen-TTS-Batch-Concurrency,X-Qwen-TTS-Audio-Seconds}
QWEN_TTS_CORS_ALLOW_CREDENTIALS=${QWEN_TTS_CORS_ALLOW_CREDENTIALS:-0}

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"
CONTAINER_NAME="qwen3-tts-api-cors-${RUN_TAG//[^a-zA-Z0-9_.-]/-}"
REMOTE_RUN_DIR="$REMOTE_RESULTS_DIR/$MACHINE_NAME/cors_regression/$RUN_TAG"
LOCAL_RUN_DIR="$REPO_ROOT/benchmarks/results/$MACHINE_NAME/cors_regression/$RUN_TAG"

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/scripts' '$REMOTE_RUN_DIR'"
scp "${SSH_OPTS[@]}" "$REPO_ROOT/scripts/qwen_tts_api_cors_regression.py" "$REMOTE_HOST:$REMOTE_ROOT/scripts/qwen_tts_api_cors_regression.py"

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
  IMAGE="$IMAGE" \
  REMOTE_ROOT="$REMOTE_ROOT" \
  REMOTE_RUN_DIR="$REMOTE_RUN_DIR" \
  CONTAINER_NAME="$CONTAINER_NAME" \
  MACHINE_NAME="$MACHINE_NAME" \
  RUN_TAG="$RUN_TAG" \
  ORIGIN="$ORIGIN" \
  REQUEST_METHOD="$REQUEST_METHOD" \
  REQUEST_HEADERS="$REQUEST_HEADERS" \
  QWEN_TTS_PORT="$QWEN_TTS_PORT" \
  QWEN_TTS_DEFAULT_SPEAKER="$QWEN_TTS_DEFAULT_SPEAKER" \
  QWEN_TTS_CORS_ALLOW_ORIGINS="$QWEN_TTS_CORS_ALLOW_ORIGINS" \
  QWEN_TTS_CORS_ALLOW_METHODS="$QWEN_TTS_CORS_ALLOW_METHODS" \
  QWEN_TTS_CORS_ALLOW_HEADERS="$QWEN_TTS_CORS_ALLOW_HEADERS" \
  QWEN_TTS_CORS_EXPOSE_HEADERS="$QWEN_TTS_CORS_EXPOSE_HEADERS" \
  QWEN_TTS_CORS_ALLOW_CREDENTIALS="$QWEN_TTS_CORS_ALLOW_CREDENTIALS" \
  'bash -s' <<'REMOTE_EOF'
set -euo pipefail

rm -f "$REMOTE_RUN_DIR/service.log"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --rm --network host --runtime=nvidia \
  --name "$CONTAINER_NAME" \
  -e QWEN_TTS_PORT="$QWEN_TTS_PORT" \
  -e QWEN_TTS_DEFAULT_SPEAKER="$QWEN_TTS_DEFAULT_SPEAKER" \
  -e QWEN_TTS_CORS_ALLOW_ORIGINS="$QWEN_TTS_CORS_ALLOW_ORIGINS" \
  -e QWEN_TTS_CORS_ALLOW_METHODS="$QWEN_TTS_CORS_ALLOW_METHODS" \
  -e QWEN_TTS_CORS_ALLOW_HEADERS="$QWEN_TTS_CORS_ALLOW_HEADERS" \
  -e QWEN_TTS_CORS_EXPOSE_HEADERS="$QWEN_TTS_CORS_EXPOSE_HEADERS" \
  -e QWEN_TTS_CORS_ALLOW_CREDENTIALS="$QWEN_TTS_CORS_ALLOW_CREDENTIALS" \
  "$IMAGE" >/dev/null

cleanup() {
  docker logs "$CONTAINER_NAME" > "$REMOTE_RUN_DIR/service.log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if curl -sSf "http://127.0.0.1:$QWEN_TTS_PORT/healthz" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "[error] service failed to become ready" >&2
  docker logs "$CONTAINER_NAME" || true
  exit 1
fi

python3 "$REMOTE_ROOT/scripts/qwen_tts_api_cors_regression.py" \
  --api-base-url "http://127.0.0.1:$QWEN_TTS_PORT" \
  --origin "$ORIGIN" \
  --request-method "$REQUEST_METHOD" \
  --request-headers "$REQUEST_HEADERS" \
  --machine-name "$MACHINE_NAME" \
  --run-tag "$RUN_TAG" \
  --output-dir "$REMOTE_RUN_DIR"
REMOTE_EOF

mkdir -p "$LOCAL_RUN_DIR"
rsync -a -e "$RSYNC_SSH" "$REMOTE_HOST:$REMOTE_RUN_DIR/" "$LOCAL_RUN_DIR/"
echo "[ok] copied results to $LOCAL_RUN_DIR"
