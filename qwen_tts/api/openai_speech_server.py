from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from qwen_tts import Qwen3TTSModel

SENTENCE_RE = r"[^。！？!?；;:\n]+[。！？!?；;:\n]?"
CLAUSE_RE = r"[^，,]+[，,]?"
SUPPORTED_RESPONSE_FORMATS = {"mp3", "wav", "flac", "opus", "aac", "pcm"}
MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "application/octet-stream",
}


@dataclass
class ServerSettings:
    model_path: str
    host: str
    port: int
    device: str
    dtype: str
    flash_attn: bool
    default_language: str
    default_speaker: str | None
    default_instruct: str
    default_response_format: str
    max_text_chars_per_segment: int
    batch_concurrency: int
    max_concurrent_requests: int
    segment_mode: str
    max_new_tokens: int | None
    voice_map: dict[str, str]
    cors_allow_origins: list[str]
    cors_allow_origin_regex: str | None
    cors_allow_methods: list[str]
    cors_allow_headers: list[str]
    cors_expose_headers: list[str]
    cors_allow_credentials: bool


class OpenAISpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str = Field(min_length=1)
    voice: str | None = None
    speaker: str | None = None
    response_format: Literal["mp3", "wav", "flac", "opus", "aac", "pcm"] | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str | None = None
    instructions: str | None = None
    instruct: str | None = None


class ModelInfoResponse(BaseModel):
    model_path: str
    tts_model_type: str
    tts_model_size: str | None
    supported_languages: list[str] | None
    supported_speakers: list[str] | None
    settings: dict[str, Any]


class TTSService:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self._model: Qwen3TTSModel | None = None
        self._model_lock = threading.Lock()
        self._request_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    def _dtype_from_name(self, name: str) -> torch.dtype:
        normalized = name.strip().lower()
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {name}")

    def _model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "device_map": self.settings.device,
            "dtype": self._dtype_from_name(self.settings.dtype),
        }
        if self.settings.flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"
        return kwargs

    def get_model(self) -> Qwen3TTSModel:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                self._model = Qwen3TTSModel.from_pretrained(self.settings.model_path, **self._model_kwargs())
        return self._model

    def _split_with_regex(self, text: str, pattern: str) -> list[str]:
        import re

        parts = [match.group(0).strip() for match in re.finditer(pattern, text) if match.group(0).strip()]
        return parts if parts else ([text.strip()] if text.strip() else [])

    def _fixed_width_chunks(self, text: str, max_chars: int) -> list[str]:
        stripped = text.strip()
        return [stripped[idx: idx + max_chars] for idx in range(0, len(stripped), max_chars) if stripped[idx: idx + max_chars].strip()]

    def _split_long_fragment(self, text: str, max_chars: int) -> list[str]:
        pieces: list[str] = []
        for clause in self._split_with_regex(text, CLAUSE_RE):
            if len(clause) <= max_chars:
                pieces.append(clause)
            else:
                pieces.extend(self._fixed_width_chunks(clause, max_chars))
        return pieces or self._fixed_width_chunks(text, max_chars)

    def split_text(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        max_chars = self.settings.max_text_chars_per_segment
        if max_chars <= 0 or len(stripped) <= max_chars:
            return [stripped]

        sentence_fragments: list[str] = []
        for sentence in self._split_with_regex(stripped, SENTENCE_RE):
            if len(sentence) <= max_chars:
                sentence_fragments.append(sentence)
            else:
                sentence_fragments.extend(self._split_long_fragment(sentence, max_chars))

        if self.settings.segment_mode == "sentence":
            return [fragment.strip() for fragment in sentence_fragments if fragment.strip()]

        segments: list[str] = []
        current = ""
        for fragment in sentence_fragments:
            fragment = fragment.strip()
            if not fragment:
                continue
            if not current:
                current = fragment
                continue
            if len(current) + len(fragment) <= max_chars:
                current += fragment
            else:
                segments.append(current)
                current = fragment
        if current:
            segments.append(current)
        return segments or [stripped]

    def _chunked(self, items: list[str], size: int) -> list[list[str]]:
        return [items[index:index + size] for index in range(0, len(items), size)]

    def _resolve_speaker(self, request: OpenAISpeechRequest) -> str:
        speaker = request.speaker or request.voice or self.settings.default_speaker
        if not speaker:
            raise HTTPException(status_code=400, detail="`voice` is required when no default speaker is configured")
        return self.settings.voice_map.get(speaker, speaker)

    def _resolve_instruction(self, request: OpenAISpeechRequest) -> str:
        return request.instruct or request.instructions or self.settings.default_instruct

    def _resolve_language(self, request: OpenAISpeechRequest) -> str:
        return request.language or self.settings.default_language

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        if abs(speed - 1.0) < 1e-6:
            return audio.astype(np.float32)
        stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=float(speed))
        return stretched.astype(np.float32)

    def _ensure_ffmpeg(self) -> None:
        if not shutil.which("ffmpeg"):
            raise HTTPException(status_code=500, detail="ffmpeg is required for mp3/opus/aac output")

    def _encode_with_ffmpeg(self, wav: np.ndarray, sample_rate: int, response_format: str) -> bytes:
        codec_args = {
            "mp3": ["-f", "mp3"],
            "opus": ["-c:a", "libopus", "-f", "ogg"],
            "aac": ["-c:a", "aac", "-f", "adts"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / f"output.{response_format}"
            sf.write(input_path, wav, sample_rate, format="WAV")
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                *codec_args[response_format],
                str(output_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise HTTPException(status_code=500, detail=f"ffmpeg encode failed: {completed.stderr.strip() or completed.stdout.strip()}")
            return output_path.read_bytes()

    def encode_audio(self, wav: np.ndarray, sample_rate: int, response_format: str) -> bytes:
        if response_format == "wav":
            with io.BytesIO() as buffer:
                sf.write(buffer, wav, sample_rate, format="WAV")
                return buffer.getvalue()
        if response_format == "flac":
            with io.BytesIO() as buffer:
                sf.write(buffer, wav, sample_rate, format="FLAC")
                return buffer.getvalue()
        if response_format == "pcm":
            pcm16 = np.clip(wav, -1.0, 1.0)
            return (pcm16 * 32767.0).astype(np.int16).tobytes()
        if response_format in {"mp3", "opus", "aac"}:
            self._ensure_ffmpeg()
            return self._encode_with_ffmpeg(wav, sample_rate, response_format)
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

    def _synthesize_sync(self, request: OpenAISpeechRequest) -> tuple[bytes, str, dict[str, Any]]:
        model = self.get_model()
        if getattr(model.model, "tts_model_type", None) != "custom_voice":
            raise HTTPException(status_code=400, detail=f"Only custom_voice models are currently supported by /v1/audio/speech, got {getattr(model.model, 'tts_model_type', None)}")

        input_text = request.input.strip()
        if not input_text:
            raise HTTPException(status_code=400, detail="`input` must not be empty")

        segments = self.split_text(input_text)
        if not segments:
            raise HTTPException(status_code=400, detail="`input` must not be empty after trimming")

        speaker = self._resolve_speaker(request)
        language = self._resolve_language(request)
        instruct = self._resolve_instruction(request)
        response_format = request.response_format or self.settings.default_response_format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

        all_wavs: list[np.ndarray] = []
        sample_rate = 0
        generate_kwargs: dict[str, Any] = {}
        if self.settings.max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = self.settings.max_new_tokens

        for batch in self._chunked(segments, self.settings.batch_concurrency):
            wavs, batch_sample_rate = model.generate_custom_voice(
                text=batch,
                speaker=[speaker] * len(batch),
                language=[language] * len(batch),
                instruct=[instruct] * len(batch),
                **generate_kwargs,
            )
            if sample_rate and sample_rate != batch_sample_rate:
                raise HTTPException(status_code=500, detail=f"Inconsistent sample rate: {sample_rate} vs {batch_sample_rate}")
            sample_rate = batch_sample_rate
            all_wavs.extend([np.asarray(wav, dtype=np.float32) for wav in wavs])

        if not all_wavs or sample_rate <= 0:
            raise HTTPException(status_code=500, detail="Model returned empty audio")

        combined = np.concatenate(all_wavs).astype(np.float32)
        combined = self._apply_speed(combined, request.speed)
        encoded = self.encode_audio(combined, sample_rate, response_format)
        metadata = {
            "segment_count": len(segments),
            "max_text_chars_per_segment": self.settings.max_text_chars_per_segment,
            "batch_concurrency": self.settings.batch_concurrency,
            "speaker": speaker,
            "language": language,
            "sample_rate": sample_rate,
            "audio_seconds": float(len(combined) / sample_rate),
        }
        return encoded, response_format, metadata

    async def synthesize(self, request: OpenAISpeechRequest) -> tuple[bytes, str, dict[str, Any]]:
        async with self._request_semaphore:
            return await asyncio.to_thread(self._synthesize_sync, request)

    def model_info(self) -> ModelInfoResponse:
        model = self.get_model()
        supported_languages = None
        if callable(getattr(model.model, "get_supported_languages", None)):
            supported_languages = model.model.get_supported_languages()
        supported_speakers = None
        if callable(getattr(model.model, "get_supported_speakers", None)):
            supported_speakers = model.get_supported_speakers()
        return ModelInfoResponse(
            model_path=self.settings.model_path,
            tts_model_type=getattr(model.model, "tts_model_type", "unknown"),
            tts_model_size=getattr(model.model, "tts_model_size", None),
            supported_languages=supported_languages,
            supported_speakers=supported_speakers,
            settings=asdict(self.settings),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default.copy()
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default.copy()


def load_settings() -> ServerSettings:
    voice_map_raw = os.environ.get("QWEN_TTS_VOICE_MAP_JSON", "")
    voice_map = json.loads(voice_map_raw) if voice_map_raw.strip() else {}
    if not isinstance(voice_map, dict):
        raise ValueError("QWEN_TTS_VOICE_MAP_JSON must decode to a JSON object")

    return ServerSettings(
        model_path=os.environ.get("QWEN_TTS_MODEL_PATH", "/opt/models/model"),
        host=os.environ.get("QWEN_TTS_HOST", "0.0.0.0"),
        port=_env_int("QWEN_TTS_PORT", 8000),
        device=os.environ.get("QWEN_TTS_DEVICE", "cuda:0"),
        dtype=os.environ.get("QWEN_TTS_DTYPE", "bfloat16"),
        flash_attn=_env_bool("QWEN_TTS_FLASH_ATTN", False),
        default_language=os.environ.get("QWEN_TTS_DEFAULT_LANGUAGE", "Auto"),
        default_speaker=os.environ.get("QWEN_TTS_DEFAULT_SPEAKER") or None,
        default_instruct=os.environ.get("QWEN_TTS_DEFAULT_INSTRUCT", ""),
        default_response_format=os.environ.get("QWEN_TTS_DEFAULT_RESPONSE_FORMAT", "mp3"),
        max_text_chars_per_segment=_env_int("QWEN_TTS_MAX_TEXT_CHARS_PER_SEGMENT", 1024),
        batch_concurrency=max(1, _env_int("QWEN_TTS_BATCH_CONCURRENCY", 1)),
        max_concurrent_requests=max(1, _env_int("QWEN_TTS_MAX_CONCURRENT_REQUESTS", 1)),
        segment_mode=os.environ.get("QWEN_TTS_SEGMENT_MODE", "sentence"),
        max_new_tokens=int(os.environ["QWEN_TTS_MAX_NEW_TOKENS"]) if os.environ.get("QWEN_TTS_MAX_NEW_TOKENS") else None,
        voice_map={str(key): str(value) for key, value in voice_map.items()},
        cors_allow_origins=_env_csv("QWEN_TTS_CORS_ALLOW_ORIGINS", ["*"]),
        cors_allow_origin_regex=os.environ.get("QWEN_TTS_CORS_ALLOW_ORIGIN_REGEX") or None,
        cors_allow_methods=_env_csv("QWEN_TTS_CORS_ALLOW_METHODS", ["*"]),
        cors_allow_headers=_env_csv("QWEN_TTS_CORS_ALLOW_HEADERS", ["*"]),
        cors_expose_headers=_env_csv(
            "QWEN_TTS_CORS_EXPOSE_HEADERS",
            [
                "X-Qwen-TTS-Segment-Count",
                "X-Qwen-TTS-Batch-Concurrency",
                "X-Qwen-TTS-Audio-Seconds",
            ],
        ),
        cors_allow_credentials=_env_bool("QWEN_TTS_CORS_ALLOW_CREDENTIALS", False),
    )


def create_app() -> FastAPI:
    settings = load_settings()
    service = TTSService(settings)
    app = FastAPI(title="Qwen-TTS OpenAI-Compatible API", version="0.1.0")
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_origin_regex=settings.cors_allow_origin_regex,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        expose_headers=settings.cors_expose_headers,
        allow_credentials=settings.cors_allow_credentials,
    )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        model_loaded = service._model is not None
        return JSONResponse({
            "status": "ok",
            "model_loaded": model_loaded,
            "model_path": settings.model_path,
        })

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        info = await asyncio.to_thread(service.model_info)
        return JSONResponse({"data": [info.model_dump()]})

    @app.post("/v1/audio/speech")
    async def create_speech(request: OpenAISpeechRequest) -> Response:
        audio_bytes, response_format, metadata = await service.synthesize(request)
        headers = {
            "X-Qwen-TTS-Segment-Count": str(metadata["segment_count"]),
            "X-Qwen-TTS-Batch-Concurrency": str(metadata["batch_concurrency"]),
            "X-Qwen-TTS-Audio-Seconds": f"{metadata['audio_seconds']:.3f}",
        }
        return Response(content=audio_bytes, media_type=MIME_TYPES[response_format], headers=headers)

    return app


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "qwen_tts.api.openai_speech_server:create_app",
        host=settings.host,
        port=settings.port,
        factory=True,
    )


if __name__ == "__main__":
    main()
