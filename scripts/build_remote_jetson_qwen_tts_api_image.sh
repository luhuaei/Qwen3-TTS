#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST=${REMOTE_HOST:-nvidia@192.168.1.230}
REMOTE_ROOT=${REMOTE_ROOT:-/home/nvidia/qwen3-tts-api}
REMOTE_BUILD_ROOT=${REMOTE_BUILD_ROOT:-$REMOTE_ROOT/build/jetson-qwen-tts-api-image}
IMAGE_NAME=${IMAGE_NAME:-qwen3-tts-api-jetson:latest}
BASE_IMAGE=${BASE_IMAGE:-registry.lazycat.cloud/x/lzc-aipod-vllm:d59c2ca}
LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-}
BUILD_ARGS=${BUILD_ARGS:-}

if [[ -z "$LOCAL_MODEL_DIR" ]]; then
  echo "[error] please set LOCAL_MODEL_DIR to a local model directory" >&2
  exit 1
fi
LOCAL_MODEL_DIR=$(cd "$LOCAL_MODEL_DIR" && pwd)

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_BUILD_ROOT/model'"
scp "${SSH_OPTS[@]}" "$REPO_ROOT/docker/jetson-qwen-tts-api.Dockerfile" "$REMOTE_HOST:$REMOTE_BUILD_ROOT/Dockerfile"
scp "${SSH_OPTS[@]}" "$REPO_ROOT/docker/jetson-qwen-tts-api-image.dockerignore" "$REMOTE_HOST:$REMOTE_BUILD_ROOT/.dockerignore"
scp "${SSH_OPTS[@]}" "$REPO_ROOT/docker/jetson-qwen-tts-api-requirements.txt" "$REMOTE_HOST:$REMOTE_BUILD_ROOT/jetson-qwen-tts-api-requirements.txt"
scp "${SSH_OPTS[@]}" "$REPO_ROOT/pyproject.toml" "$REMOTE_HOST:$REMOTE_BUILD_ROOT/pyproject.toml"
scp "${SSH_OPTS[@]}" "$REPO_ROOT/README.md" "$REMOTE_HOST:$REMOTE_BUILD_ROOT/README.md"
scp "${SSH_OPTS[@]}" "$REPO_ROOT/API.md" "$REMOTE_HOST:$REMOTE_BUILD_ROOT/API.md"
rsync -a --delete -e "$RSYNC_SSH" "$REPO_ROOT/qwen_tts/" "$REMOTE_HOST:$REMOTE_BUILD_ROOT/qwen_tts/"
rsync -a --delete -e "$RSYNC_SSH" "$LOCAL_MODEL_DIR/" "$REMOTE_HOST:$REMOTE_BUILD_ROOT/model/"

PROXY_BUILD_ARGS=()
for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
  if [[ -n "${!proxy_var:-}" ]]; then
    PROXY_BUILD_ARGS+=(--build-arg "$proxy_var=${!proxy_var}")
  fi
done

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
  IMAGE_NAME="$IMAGE_NAME" \
  BASE_IMAGE="$BASE_IMAGE" \
  REMOTE_BUILD_ROOT="$REMOTE_BUILD_ROOT" \
  BUILD_ARGS="$BUILD_ARGS" \
  PROXY_BUILD_ARGS="${PROXY_BUILD_ARGS[*]}" \
  'bash -s' <<'REMOTE_EOF'
set -euo pipefail
export DOCKER_BUILDKIT=1
cd "$REMOTE_BUILD_ROOT"
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  $PROXY_BUILD_ARGS \
  -t "$IMAGE_NAME" \
  $BUILD_ARGS \
  .
REMOTE_EOF

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "docker image inspect '$IMAGE_NAME' --format 'built {{.RepoTags}} size={{.Size}}'"
