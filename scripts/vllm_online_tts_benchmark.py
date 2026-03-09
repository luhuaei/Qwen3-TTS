from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import (  # noqa: E402
    BenchmarkResult,
    append_jsonl,
    build_text_verification,
    detect_system_info,
    ensure_dir,
    strip_asr_special_tokens,
    transcribe_audio_openai_compat,
    write_json,
)


DEFAULT_TEXT = """临海的小城入秋总比别处慢半拍。清晨的风从骑楼缝隙里穿过去，带着一点潮湿，也带着刚出炉面包的甜味。她沿着旧街往前走，鞋跟轻轻敲在石板路上，像在替这座城市数着缓慢的心跳。街角书店的老板正把木门推开，门铃响了一下，惊醒了窗边打盹的猫。远处有人在摊位前挑选橘子，讨价还价的声音不高，却让整条街显得更有人情味。她忽然觉得，生活其实并不急着给出答案，只要你愿意停下来听一听，连风声里都藏着温柔的回音。"""
SENTENCE_RE = re.compile(r"[^。！？!?；;:\n]+[。！？!?；;:\n]?")
CLAUSE_RE = re.compile(r"[^，,]+[，,]?")
LANGUAGE_TO_ASR_CODE = {
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "german": "de",
    "french": "fr",
    "russian": "ru",
    "portuguese": "pt",
    "spanish": "es",
    "italian": "it",
}
DEFAULT_STREAM_SAMPLE_RATE = 24000


@dataclass
class RequestResult:
    index: int
    text: str
    status_code: int
    duration_seconds: float
    audio_seconds: float
    output_path: str
    content_type: str
    headers: dict[str, str]
    asr: dict[str, Any]
    verification: dict[str, Any]


@dataclass
class StreamResult:
    status_code: int
    duration_seconds: float
    first_chunk_seconds: float | None
    chunk_count: int
    byte_count: int
    output_path: str
    audio_seconds: float
    content_type: str
    headers: dict[str, str]
    asr: dict[str, Any]
    verification: dict[str, Any]
    streaming_supported: bool


@dataclass
class VoicesResult:
    status_code: int
    voices: list[str]
    requested_voice_present: bool
    headers: dict[str, str]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-TTS through vLLM OpenAI-compatible speech endpoints.")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--request-model-name", default=None)
    parser.add_argument("--machine-name", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--startup-seconds", type=float, default=0.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--task-type", default="CustomVoice")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output-root", default="/workspace/benchmarks/results")
    parser.add_argument("--max-text-chars-per-segment", type=int, default=256)
    parser.add_argument("--segment-mode", default="sentence", choices=["sentence", "packed"])
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--concurrency-candidates", default="1,2,4,8,16")
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--stream-read-size", type=int, default=4096)
    parser.add_argument("--stream-initial-codec-chunk-frames", type=int, default=6)
    parser.add_argument("--asr-base-url", default=os.environ.get("ASR_BASE_URL", "http://192.168.1.230:10001"))
    parser.add_argument("--asr-model", default=os.environ.get("ASR_MODEL", "sensevoice-small"))
    parser.add_argument("--asr-timeout", type=int, default=int(os.environ.get("ASR_TIMEOUT", "300")))
    parser.add_argument("--verify-min-length-ratio", type=float, default=float(os.environ.get("VERIFY_MIN_LENGTH_RATIO", "0.75")))
    parser.add_argument("--verify-min-coverage-ratio", type=float, default=float(os.environ.get("VERIFY_MIN_COVERAGE_RATIO", "0.65")))
    return parser.parse_args()



def resolve_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return args.text.strip()



def parse_positive_int_candidates(raw: str, default_value: int) -> list[int]:
    values: list[int] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value > 0 and value not in values:
            values.append(value)
    if default_value > 0 and default_value not in values:
        values.insert(0, default_value)
    return values or [default_value]



def _split_with_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    parts = [match.group(0).strip() for match in pattern.finditer(text) if match.group(0).strip()]
    stripped = text.strip()
    return parts if parts else ([stripped] if stripped else [])



def _fixed_width_chunks(text: str, max_chars: int) -> list[str]:
    stripped = text.strip()
    return [stripped[idx: idx + max_chars] for idx in range(0, len(stripped), max_chars) if stripped[idx: idx + max_chars].strip()]



def _split_long_fragment(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for clause in _split_with_pattern(text, CLAUSE_RE):
        if len(clause) <= max_chars:
            pieces.append(clause)
        else:
            pieces.extend(_fixed_width_chunks(clause, max_chars))
    return pieces or _fixed_width_chunks(text, max_chars)



def split_text_into_segments(text: str, max_chars: int, mode: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if max_chars <= 0 or len(stripped) <= max_chars:
        return [stripped]

    sentence_fragments: list[str] = []
    for sentence in _split_with_pattern(stripped, SENTENCE_RE):
        if len(sentence) <= max_chars:
            sentence_fragments.append(sentence)
        else:
            sentence_fragments.extend(_split_long_fragment(sentence, max_chars))

    if mode == "sentence":
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



def prepare_request_texts(text: str, max_chars: int, mode: str, minimum_count: int) -> list[str]:
    segments = split_text_into_segments(text, max_chars, mode)
    if not segments:
        return []
    requests = list(segments)
    index = 0
    while len(requests) < minimum_count:
        requests.append(segments[index % len(segments)])
        index += 1
    return requests



def infer_asr_language(language: str) -> str | None:
    return LANGUAGE_TO_ASR_CODE.get(language.strip().lower())



def resolve_request_model_name(args: argparse.Namespace) -> str:
    return (args.request_model_name or args.model_name).strip()


def _header_dict(headers: Any) -> dict[str, str]:
    if hasattr(headers, "items"):
        return {str(k): str(v) for k, v in headers.items()}
    return {str(k): str(v) for k, v in dict(headers).items()}



def _http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: int) -> tuple[int, dict[str, str], bytes]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _header_dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, _header_dict(exc.headers), exc.read()



def _post_audio(api_base_url: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, str], bytes]:
    return _http_json("POST", api_base_url.rstrip("/") + "/v1/audio/speech", payload, timeout)



def _get_voices(api_base_url: str, timeout: int) -> tuple[int, dict[str, str], bytes]:
    return _http_json("GET", api_base_url.rstrip("/") + "/v1/audio/voices", None, timeout)



def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sample_rate = wf.getframerate()
    return float(frames / sample_rate) if sample_rate > 0 else 0.0



def write_pcm16_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)



def verify_audio_file(args: argparse.Namespace, wav_path: Path, expected_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    asr_payload = transcribe_audio_openai_compat(
        wav_path,
        base_url=args.asr_base_url,
        model=args.asr_model,
        timeout=args.asr_timeout,
        language=infer_asr_language(args.language),
        response_format="json",
    )
    cleaned_text = strip_asr_special_tokens(asr_payload.get("text") or asr_payload.get("result") or "")
    if "text" in asr_payload:
        asr_payload["text"] = cleaned_text
    if "result" in asr_payload:
        asr_payload["result"] = cleaned_text
    verification = build_text_verification(
        expected_text,
        cleaned_text,
        min_length_ratio=args.verify_min_length_ratio,
        min_matching_ratio=args.verify_min_coverage_ratio,
    )
    return asr_payload, verification



def run_voices_regression(args: argparse.Namespace, run_dir: Path) -> VoicesResult:
    status_code, headers, body = _get_voices(args.api_base_url, args.request_timeout)
    payload = json.loads(body.decode("utf-8", errors="replace")) if body else {}
    voices = [str(item) for item in payload.get("voices", [])]
    result = VoicesResult(
        status_code=status_code,
        voices=voices,
        requested_voice_present=args.voice.lower() in {voice.lower() for voice in voices} if voices else False,
        headers=headers,
    )
    write_json(run_dir / "voices_result.json", asdict(result))
    return result



def run_non_stream_regression(args: argparse.Namespace, run_dir: Path, text: str) -> RequestResult:
    payload: dict[str, Any] = {
        "model": resolve_request_model_name(args),
        "input": text,
        "voice": args.voice,
        "language": args.language,
        "instructions": args.instructions,
        "task_type": args.task_type,
        "response_format": "wav",
        "max_new_tokens": args.max_new_tokens,
    }
    started = time.perf_counter()
    status_code, headers, body = _post_audio(args.api_base_url, payload, args.request_timeout)
    duration_seconds = time.perf_counter() - started
    output_path = run_dir / "regression.wav"
    output_path.write_bytes(body)
    if status_code == 200:
        audio_seconds = wav_duration_seconds(output_path)
        asr_payload, verification = verify_audio_file(args, output_path, text)
    else:
        audio_seconds = 0.0
        asr_payload = {"error": output_path.read_text(encoding="utf-8", errors="replace")}
        verification = build_text_verification(
            text,
            "",
            min_length_ratio=args.verify_min_length_ratio,
            min_matching_ratio=args.verify_min_coverage_ratio,
        )
    result = RequestResult(
        index=0,
        text=text,
        status_code=status_code,
        duration_seconds=duration_seconds,
        audio_seconds=audio_seconds,
        output_path=str(output_path),
        content_type=headers.get("Content-Type", headers.get("content-type", "")),
        headers=headers,
        asr=asr_payload,
        verification=verification,
    )
    write_json(run_dir / "regression_result.json", asdict(result))
    return result



def run_stream_regression(args: argparse.Namespace, run_dir: Path, text: str) -> StreamResult:
    payload: dict[str, Any] = {
        "model": resolve_request_model_name(args),
        "input": text,
        "voice": args.voice,
        "language": args.language,
        "instructions": args.instructions,
        "task_type": args.task_type,
        "response_format": "pcm",
        "stream": True,
        "speed": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "initial_codec_chunk_frames": args.stream_initial_codec_chunk_frames,
    }
    req = urllib.request.Request(
        args.api_base_url.rstrip("/") + "/v1/audio/speech",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    output_path = run_dir / "stream_regression.wav"
    started = time.perf_counter()
    chunk_count = 0
    byte_count = 0
    first_chunk_seconds: float | None = None
    headers: dict[str, str] = {}
    status_code = 0
    body_parts: list[bytes] = []
    with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
        status_code = resp.status
        headers = _header_dict(resp.headers)
        while True:
            chunk = resp.read(args.stream_read_size)
            if not chunk:
                break
            chunk_count += 1
            byte_count += len(chunk)
            body_parts.append(chunk)
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - started
    duration_seconds = time.perf_counter() - started
    raw_audio = b"".join(body_parts)
    if status_code == 200:
        write_pcm16_wav(output_path, raw_audio, DEFAULT_STREAM_SAMPLE_RATE)
        audio_seconds = wav_duration_seconds(output_path)
        asr_payload, verification = verify_audio_file(args, output_path, text)
    else:
        output_path.write_bytes(raw_audio)
        audio_seconds = 0.0
        asr_payload = {"error": output_path.read_text(encoding="utf-8", errors="replace")}
        verification = build_text_verification(
            text,
            "",
            min_length_ratio=args.verify_min_length_ratio,
            min_matching_ratio=args.verify_min_coverage_ratio,
        )
    streaming_supported = bool(status_code == 200 and chunk_count > 0 and first_chunk_seconds is not None and first_chunk_seconds < duration_seconds)
    result = StreamResult(
        status_code=status_code,
        duration_seconds=duration_seconds,
        first_chunk_seconds=first_chunk_seconds,
        chunk_count=chunk_count,
        byte_count=byte_count,
        output_path=str(output_path),
        audio_seconds=audio_seconds,
        content_type=headers.get("Content-Type", headers.get("content-type", "")),
        headers=headers,
        asr=asr_payload,
        verification=verification,
        streaming_supported=streaming_supported,
    )
    write_json(run_dir / "stream_result.json", asdict(result))
    return result



def run_single_request(args: argparse.Namespace, candidate_dir: Path, index: int, text: str) -> RequestResult:
    payload: dict[str, Any] = {
        "model": resolve_request_model_name(args),
        "input": text,
        "voice": args.voice,
        "language": args.language,
        "instructions": args.instructions,
        "task_type": args.task_type,
        "response_format": "wav",
        "max_new_tokens": args.max_new_tokens,
    }
    started = time.perf_counter()
    status_code, headers, body = _post_audio(args.api_base_url, payload, args.request_timeout)
    duration_seconds = time.perf_counter() - started
    output_path = candidate_dir / f"output_{index:03d}.wav"
    output_path.write_bytes(body)
    if status_code == 200:
        audio_seconds = wav_duration_seconds(output_path)
        asr_payload, verification = verify_audio_file(args, output_path, text)
    else:
        audio_seconds = 0.0
        asr_payload = {"error": output_path.read_text(encoding="utf-8", errors="replace")}
        verification = build_text_verification(
            text,
            "",
            min_length_ratio=args.verify_min_length_ratio,
            min_matching_ratio=args.verify_min_coverage_ratio,
        )
    return RequestResult(
        index=index,
        text=text,
        status_code=status_code,
        duration_seconds=duration_seconds,
        audio_seconds=audio_seconds,
        output_path=str(output_path),
        content_type=headers.get("Content-Type", headers.get("content-type", "")),
        headers=headers,
        asr=asr_payload,
        verification=verification,
    )



def write_candidate_report(candidate_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Benchmark Report",
        "",
        f"- `concurrency`: {payload['concurrency']}",
        f"- `request_count`: {payload['request_count']}",
        f"- `startup_seconds`: {payload['startup_seconds']:.3f}",
        f"- `request_seconds`: {payload['request_seconds']:.3f}",
        f"- `total_audio_seconds`: {payload['audio_seconds']:.3f}",
        f"- `request_x_realtime`: {payload['request_x_realtime']:.3f}",
        f"- `avg_latency_seconds`: {payload['avg_latency_seconds']:.3f}",
        f"- `p95_latency_seconds`: {payload['p95_latency_seconds']:.3f}",
        f"- `verification_passed`: {payload['verification_passed']}",
        "",
        "## Requests",
        "",
        "| index | status | latency_s | audio_s | passed | coverage | length | output |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for item in payload["requests"]:
        verification = item["verification"]
        lines.append(
            "| {index} | {status} | {latency:.3f} | {audio:.3f} | {passed} | {coverage:.3f} | {length:.3f} | `{output}` |".format(
                index=item["index"],
                status=item["status_code"],
                latency=item["duration_seconds"],
                audio=item["audio_seconds"],
                passed=verification["passed"],
                coverage=verification["coverage_ratio"],
                length=verification["length_ratio"],
                output=Path(item["output_path"]).name,
            )
        )
    (candidate_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])



def run_concurrency_candidate(args: argparse.Namespace, run_dir: Path, startup_seconds: float, request_texts: list[str], concurrency: int) -> dict[str, Any]:
    candidate_dir = ensure_dir(run_dir / f"conc-{concurrency}")
    started = time.perf_counter()
    results: list[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_single_request, args, candidate_dir, index, text)
            for index, text in enumerate(request_texts)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    request_seconds = time.perf_counter() - started
    results.sort(key=lambda item: item.index)

    total_audio_seconds = sum(item.audio_seconds for item in results)
    latencies = [item.duration_seconds for item in results]
    verification_passed = all(item.verification["passed"] for item in results)
    payload = {
        "benchmark_name": "qwen3_tts_vllm_openai_online",
        "backend": "remote_vllm_endpoint",
        "machine_name": args.machine_name,
        "model_name": args.model_name,
        "request_model_name": resolve_request_model_name(args),
        "run_tag": args.run_tag,
        "concurrency": concurrency,
        "request_count": len(request_texts),
        "startup_seconds": startup_seconds,
        "request_seconds": request_seconds,
        "request_available_after_seconds": startup_seconds,
        "audio_seconds": total_audio_seconds,
        "request_rtf": request_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0,
        "request_x_realtime": total_audio_seconds / request_seconds if request_seconds > 0 else 0.0,
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "verification_passed": verification_passed,
        "requests": [asdict(item) for item in results],
    }
    write_json(candidate_dir / "candidate_result.json", payload)
    write_candidate_report(candidate_dir, payload)
    return payload



def choose_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [candidate for candidate in candidates if candidate["verification_passed"]]
    pool = passed or candidates
    return max(pool, key=lambda item: (int(item["verification_passed"]), item["request_x_realtime"], -item["p95_latency_seconds"]))



def write_top_level_report(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# vLLM Online TTS Benchmark",
        "",
        f"- `machine_name`: {payload['machine_name']}",
        f"- `model_name`: {payload['model_name']}",
        f"- `startup_seconds`: {payload['startup_seconds']:.3f}",
        f"- `voice_count`: {len(payload['voices']['voices'])}",
        f"- `requested_voice_present`: {payload['voices']['requested_voice_present']}",
        f"- `speech_regression_passed`: {payload['regression']['verification']['passed']}",
        f"- `streaming_supported`: {payload['streaming']['streaming_supported']}",
        f"- `stream_first_chunk_seconds`: {payload['streaming']['first_chunk_seconds'] if payload['streaming']['first_chunk_seconds'] is not None else 'n/a'}",
        "",
        "## Concurrency Candidates",
        "",
        "| concurrency | request_count | request_seconds | total_audio_seconds | x_realtime | avg_latency_s | p95_latency_s | passed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in payload["tuning"]["candidates"]:
        lines.append(
            "| {concurrency} | {request_count} | {request_seconds:.3f} | {audio_seconds:.3f} | {request_x_realtime:.3f} | {avg_latency_seconds:.3f} | {p95_latency_seconds:.3f} | {verification_passed} |".format(**candidate)
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    args = parse_args()
    text = resolve_text(args)
    machine_dir = ensure_dir(Path(args.output_root) / args.machine_name / "remote_vllm" / args.run_tag)

    voices_result = run_voices_regression(args, machine_dir)
    if voices_result.status_code != 200:
        raise SystemExit(f"voices endpoint failed with status {voices_result.status_code}")
    if args.task_type == "CustomVoice" and not voices_result.requested_voice_present:
        raise SystemExit(f"voice '{args.voice}' not found in /v1/audio/voices: {voices_result.voices}")

    regression_result = run_non_stream_regression(args, machine_dir, text)
    if regression_result.status_code != 200:
        raise SystemExit(f"speech endpoint regression failed with status {regression_result.status_code}")
    if not regression_result.verification["passed"]:
        raise SystemExit(f"speech regression ASR verification failed: {regression_result.verification['reason']}")

    stream_result = run_stream_regression(args, machine_dir, text)
    if stream_result.status_code != 200:
        raise SystemExit(f"stream regression failed with status {stream_result.status_code}")
    if not stream_result.verification["passed"]:
        raise SystemExit(f"stream regression ASR verification failed: {stream_result.verification['reason']}")

    concurrency_candidates = parse_positive_int_candidates(args.concurrency_candidates, args.concurrency)
    candidate_results: list[dict[str, Any]] = []
    for concurrency in concurrency_candidates:
        request_texts = prepare_request_texts(
            text,
            args.max_text_chars_per_segment,
            args.segment_mode,
            minimum_count=concurrency,
        )
        candidate_results.append(
            run_concurrency_candidate(
                args,
                machine_dir,
                args.startup_seconds,
                request_texts,
                concurrency,
            )
        )

    best_candidate = choose_best_candidate(candidate_results)
    result = BenchmarkResult(
        benchmark_name="qwen3_tts_vllm_openai_online",
        backend="remote_vllm_endpoint",
        machine_name=args.machine_name,
        model_name=args.model_name,
        model_source=args.model_name,
        prompt_count=best_candidate["request_count"],
        measured_seconds=best_candidate["request_seconds"],
        load_seconds=args.startup_seconds,
        audio_seconds=best_candidate["audio_seconds"],
        realtime_factor=best_candidate["request_rtf"],
        x_realtime=best_candidate["request_x_realtime"],
        sample_rate=DEFAULT_STREAM_SAMPLE_RATE,
        output_wavs=best_candidate["request_count"],
        output_dir=str(machine_dir),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        system_info=detect_system_info(),
        extra={
            "language": args.language,
            "voice": args.voice,
            "task_type": args.task_type,
            "text_length": len(text),
            "max_text_chars_per_segment": args.max_text_chars_per_segment,
            "segment_mode": args.segment_mode,
            "startup_seconds": args.startup_seconds,
        },
    )
    payload = result.to_dict()
    payload.update(
        {
            "startup_seconds": args.startup_seconds,
            "voices": asdict(voices_result),
            "regression": asdict(regression_result),
            "streaming": asdict(stream_result),
            "verification_passed": bool(
                regression_result.verification["passed"]
                and stream_result.verification["passed"]
                and best_candidate["verification_passed"]
            ),
            "selected_concurrency": best_candidate["concurrency"],
            "selected_candidate": best_candidate,
            "tuning": {
                "concurrency_candidates": concurrency_candidates,
                "selected_concurrency": best_candidate["concurrency"],
                "candidates": candidate_results,
            },
        }
    )
    write_json(machine_dir / "result.json", payload)
    append_jsonl(Path(args.output_root) / "benchmark_history.jsonl", payload)
    write_top_level_report(machine_dir, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if not payload["verification_passed"]:
        raise SystemExit("[error] verification failed")
    if 16 in concurrency_candidates and not any(int(candidate.get("concurrency", -1)) == 16 for candidate in candidate_results):
        raise SystemExit("[error] concurrency 16 benchmark was not executed")


if __name__ == "__main__":
    main()
