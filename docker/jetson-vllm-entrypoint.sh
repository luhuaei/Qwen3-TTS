#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

resolve_model_name() {
  if [[ -n "${MODEL_NAME:-}" ]]; then
    if [[ "$MODEL_NAME" == /* ]] && [[ ! -e "$MODEL_NAME" ]]; then
      echo "[error] MODEL_NAME points to a missing local path: $MODEL_NAME" >&2
      exit 64
    fi
    echo "$MODEL_NAME"
    return 0
  fi

  local candidates=()
  if [[ -n "${QWEN_TTS_MODEL_PATH:-}" ]]; then
    candidates+=("$QWEN_TTS_MODEL_PATH")
  fi
  candidates+=(
    "/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    "/opt/models/model"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -e "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  if [[ "${HF_HUB_OFFLINE:-0}" == "1" || "${TRANSFORMERS_OFFLINE:-0}" == "1" ]]; then
    echo "[error] no local model found for offline startup." >&2
    echo "[error] mount a model to /workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice or set MODEL_NAME=/path/to/local/model" >&2
    exit 64
  fi

  echo "${MODEL_NAME_FALLBACK:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
}

MODEL_NAME=$(resolve_model_name)
STAGE_CONFIG_PATH=${VLLM_STAGE_CONFIG_PATH:-/opt/vllm-omni/stage_configs/qwen3_tts_jetson.yaml}
HOST=${VLLM_HOST:-0.0.0.0}
PORT=${VLLM_PORT:-8091}
SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}
EXTRA_ARGS=${VLLM_SERVE_ARGS:-}

SERVED_MODEL_ARG=()
if [[ -n "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_ARG=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ ! -e "$STAGE_CONFIG_PATH" ]]; then
  echo "[error] VLLM stage config not found: $STAGE_CONFIG_PATH" >&2
  exit 64
fi

serve_args=(
  --host "$HOST"
  --port "$PORT"
  --stage-configs-path "$STAGE_CONFIG_PATH"
  --trust-remote-code
  --enforce-eager
)
if [[ -n "$SERVED_MODEL_NAME" ]]; then
  serve_args+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra_args_array <<< "$EXTRA_ARGS"
  serve_args+=("${extra_args_array[@]}")
fi

if command -v vllm >/dev/null 2>&1; then
  exec vllm serve "$MODEL_NAME" --omni "${serve_args[@]}"
fi

exec vllm-omni serve "$MODEL_NAME" --omni "${serve_args[@]}"
