# Qwen-TTS API

## Overview

本服务提供 OpenAI 兼容的 `POST /v1/audio/speech` 接口，默认面向 `Qwen3-TTS-*-CustomVoice` 模型。

特性：

- 兼容 OpenAI `audio/speech` 请求格式。
- 支持长文本自动分段并拼接音频。
- 分段最大长度和分段批并发通过环境变量控制。
- 适合在 Jetson 上打包成离线可运行镜像。

## Endpoints

### `GET /healthz`

返回服务健康状态。

### `GET /v1/models`

返回当前已加载模型和服务配置。

### `POST /v1/audio/speech`

OpenAI 兼容 TTS 接口。

请求体字段：

- `input`: 要合成的文本。
- `voice`: OpenAI 风格 voice 名称；本服务会直接映射到 Qwen-TTS speaker。
- `speaker`: 可选，若提供则优先于 `voice`。
- `response_format`: 支持 `mp3`、`wav`、`flac`、`opus`、`aac`、`pcm`。
- `speed`: 语速，范围 `0.25 ~ 4.0`，默认 `1.0`。
- `language`: 语言，可选。
- `instructions` / `instruct`: 风格指令，可选。
- `model`: 兼容字段，当前不会覆盖服务启动时加载的模型。

示例：

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-tts",
    "input": "临海的小城入秋总比别处慢半拍。",
    "voice": "Vivian",
    "response_format": "wav",
    "speed": 1.0,
    "language": "Chinese"
  }' \
  --output speech.wav
```

响应头会额外附带：

- `X-Qwen-TTS-Segment-Count`
- `X-Qwen-TTS-Batch-Concurrency`
- `X-Qwen-TTS-Audio-Seconds`

## Environment Variables

### Model and server

- `QWEN_TTS_MODEL_PATH`: 模型目录，默认 `/opt/models/model`
- `QWEN_TTS_HOST`: 监听地址，默认 `0.0.0.0`
- `QWEN_TTS_PORT`: 监听端口，默认 `8000`
- `QWEN_TTS_DEVICE`: 模型设备，默认 `cuda:0`
- `QWEN_TTS_DTYPE`: `bfloat16` / `float16` / `float32`，默认 `bfloat16`
- `QWEN_TTS_FLASH_ATTN`: 是否启用 `flash_attention_2`，默认 `0`

### Request defaults

- `QWEN_TTS_DEFAULT_SPEAKER`: 默认 speaker；请求里不传 `voice`/`speaker` 时使用
- `QWEN_TTS_DEFAULT_LANGUAGE`: 默认语言，默认 `Auto`
- `QWEN_TTS_DEFAULT_INSTRUCT`: 默认风格指令，默认空字符串
- `QWEN_TTS_DEFAULT_RESPONSE_FORMAT`: 默认输出格式，默认 `mp3`
- `QWEN_TTS_VOICE_MAP_JSON`: JSON 对象，用于把 OpenAI `voice` 映射到实际 speaker，例如 `{"alloy":"Vivian"}`

### Segmentation and concurrency

- `QWEN_TTS_MAX_TEXT_CHARS_PER_SEGMENT`: 单段最大字符数，默认 `1024`
- `QWEN_TTS_SEGMENT_MODE`: `sentence` 或 `packed`，默认 `sentence`
- `QWEN_TTS_BATCH_CONCURRENCY`: 单次模型 batch 内并行处理的分段数，默认 `1`
- `QWEN_TTS_MAX_CONCURRENT_REQUESTS`: 服务同时处理的请求数，默认 `1`
- `QWEN_TTS_MAX_NEW_TOKENS`: 透传给生成器的 `max_new_tokens`

## Build an offline Jetson image

先准备本地模型目录，例如：

- `models/Qwen3-TTS-12Hz-0.6B-CustomVoice`

然后执行远端构建：

```bash
LOCAL_MODEL_DIR=models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
IMAGE_NAME=qwen3-tts-api-jetson:latest \
bash scripts/build_remote_jetson_qwen_tts_api_image.sh
```

说明：

- 构建发生在 Jetson 远端 `ssh nvidia@192.168.1.230`。
- 镜像会内置模型目录，因此构建完成后可离线运行。
- 依赖层、模型层、代码层分开，尽量减少重复长时间构建。

## Run the offline image on Jetson

```bash
docker run --rm --network host --runtime=nvidia \
  -e QWEN_TTS_DEFAULT_SPEAKER=Vivian \
  -e QWEN_TTS_DEFAULT_LANGUAGE=Chinese \
  -e QWEN_TTS_MAX_TEXT_CHARS_PER_SEGMENT=1024 \
  -e QWEN_TTS_BATCH_CONCURRENCY=8 \
  -e QWEN_TTS_MAX_CONCURRENT_REQUESTS=1 \
  qwen3-tts-api-jetson:latest
```

启动后可访问：

- `http://<jetson-ip>:8000/docs`
- `http://<jetson-ip>:8000/v1/audio/speech`

## API ASR regression

可直接在本地发起 Jetson 远端 API + ASR 自动回归：

```bash
IMAGE=qwen3-tts-api-jetson:latest \
MACHINE_NAME=jetson-agx-orin \
RUN_TAG=api-asr-regression \
VOICE=Vivian \
TTS_LANGUAGE=Chinese \
QWEN_TTS_MAX_TEXT_CHARS_PER_SEGMENT=1024 \
QWEN_TTS_BATCH_CONCURRENCY=1 \
ASR_BASE_URL=http://192.168.1.230:10001 \
bash benchmarks/run_remote_qwen_tts_api_regression.sh
```

输出目录：

- `benchmarks/results/<machine>/api_regression/<run-tag>/result.json`
- `benchmarks/results/<machine>/api_regression/<run-tag>/report.md`
- `benchmarks/results/<machine>/api_regression/<run-tag>/output_0.wav`
- `benchmarks/results/<machine>/api_regression/<run-tag>/service.log`

## Notes

- 当前 `POST /v1/audio/speech` 主要面向 `CustomVoice` 模型。
- 若要启用 FlashAttention，请确保 Jetson 镜像内已具备可用的 `flash-attn` 运行环境，再设置 `QWEN_TTS_FLASH_ATTN=1`。
- 较长文本会自动按句切分；如句子过长，会继续按子句和固定宽度拆分后再拼接输出。
