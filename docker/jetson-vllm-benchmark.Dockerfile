ARG BASE_IMAGE=registry.lazycat.cloud/x/lzc-aipod-vllm:d59c2ca
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VLLM_OMNI_VENV=/opt/vllm-omni-venv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev0 \
    PATH=/opt/vllm-omni-venv/bin:${PATH}

WORKDIR /opt/build

RUN apt-get update \
    && apt-get install -y --no-install-recommends sox \
    && rm -rf /var/lib/apt/lists/*

COPY jetson-vllm-omni-requirements.txt /tmp/jetson-vllm-omni-requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv "${VLLM_OMNI_VENV}" --python 3.10 --system-site-packages \
    && uv pip install --python "${VLLM_OMNI_VENV}/bin/python3" -r /tmp/jetson-vllm-omni-requirements.txt \
    && uv pip install --python "${VLLM_OMNI_VENV}/bin/python3" flash-attn -i http://wa.lan:10608/simple --trusted-host wa.lan

COPY vllm-omni /opt/vllm-omni
COPY vllm_qwen3_tts_jetson_stage_config.yaml /opt/vllm-omni/stage_configs/qwen3_tts_jetson.yaml
COPY jetson-vllm-entrypoint.sh /usr/local/bin/jetson-vllm-entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VLLM_OMNI_VENV}/bin/python3" -e /opt/vllm-omni --no-deps --no-build-isolation \
    && find /opt/vllm-omni -name '__pycache__' -type d -prune -exec rm -rf '{}' +

WORKDIR /workspace
EXPOSE 8091
ENTRYPOINT ["/usr/local/bin/jetson-vllm-entrypoint.sh"]
