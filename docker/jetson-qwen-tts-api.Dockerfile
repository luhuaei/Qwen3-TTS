ARG BASE_IMAGE=registry.lazycat.cloud/x/lzc-aipod-vllm:d59c2ca
FROM scratch AS model_stage
COPY model/ /opt/models/model/

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    QWEN_TTS_VENV=/opt/qwen-tts-venv \
    PATH=/opt/qwen-tts-venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    QWEN_TTS_MODEL_PATH=/opt/models/model \
    QWEN_TTS_HOST=0.0.0.0 \
    QWEN_TTS_PORT=8000

WORKDIR /opt/build

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

COPY jetson-qwen-tts-api-requirements.txt /tmp/jetson-qwen-tts-api-requirements.txt
RUN uv venv "${QWEN_TTS_VENV}" --python 3.10 --system-site-packages \
    && uv pip install --python "${QWEN_TTS_VENV}/bin/python3" -r /tmp/jetson-qwen-tts-api-requirements.txt \
    && rm -rf /root/.cache/uv

COPY pyproject.toml README.md API.md /opt/build/
COPY --from=model_stage /opt/models/model /opt/models/model
COPY qwen_tts /opt/build/qwen_tts

RUN uv pip install --python "${QWEN_TTS_VENV}/bin/python3" -e /opt/build --no-deps --no-build-isolation \
    && find /opt/build -name '__pycache__' -type d -prune -exec rm -rf '{}' + \
    && rm -rf /root/.cache/uv

WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["qwen-tts-api"]
