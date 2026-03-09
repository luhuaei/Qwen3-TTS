from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


@dataclass
class CheckResult:
    name: str
    passed: bool
    status: int
    headers: dict[str, str]
    detail: str


def _request(method: str, url: str, headers: dict[str, str], body: bytes | None = None) -> tuple[int, dict[str, str], str]:
    req = request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=15) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.headers.items()), payload
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        return exc.code, dict(exc.headers.items()), payload


def _header_ci(headers: dict[str, str], name: str) -> str:
    lookup = name.lower()
    for key, value in headers.items():
        if key.lower() == lookup:
            return value
    return ""


def _split_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _check_preflight(api_base_url: str, origin: str, request_method: str, request_headers: str) -> CheckResult:
    status, headers, _ = _request(
        "OPTIONS",
        f"{api_base_url.rstrip('/')}/v1/audio/speech",
        {
            "Origin": origin,
            "Access-Control-Request-Method": request_method,
            "Access-Control-Request-Headers": request_headers,
        },
    )
    allow_origin = _header_ci(headers, "Access-Control-Allow-Origin")
    allow_methods = _split_csv(_header_ci(headers, "Access-Control-Allow-Methods"))
    allow_headers = _split_csv(_header_ci(headers, "Access-Control-Allow-Headers"))
    requested_headers = _split_csv(request_headers)
    problems: list[str] = []
    if status not in {200, 204}:
        problems.append(f"unexpected status={status}")
    if allow_origin not in {origin, "*"}:
        problems.append(f"allow-origin={allow_origin!r}")
    if "*" not in allow_methods and request_method.lower() not in allow_methods:
        problems.append(f"allow-methods={allow_methods}")
    missing_headers = [header for header in requested_headers if "*" not in allow_headers and header not in allow_headers]
    if missing_headers:
        problems.append(f"missing allow-headers={missing_headers}")
    return CheckResult(
        name="preflight",
        passed=not problems,
        status=status,
        headers=headers,
        detail="ok" if not problems else "; ".join(problems),
    )


def _check_post_error_path(api_base_url: str, origin: str) -> CheckResult:
    status, headers, payload = _request(
        "POST",
        f"{api_base_url.rstrip('/')}/v1/audio/speech",
        {
            "Origin": origin,
            "Content-Type": "application/json",
        },
        body=b'{"input":""}',
    )
    allow_origin = _header_ci(headers, "Access-Control-Allow-Origin")
    problems: list[str] = []
    if status not in {400, 422}:
        problems.append(f"unexpected status={status}")
    if allow_origin not in {origin, "*"}:
        problems.append(f"allow-origin={allow_origin!r}")
    if not payload.strip():
        problems.append("empty response body")
    return CheckResult(
        name="post_error_path",
        passed=not problems,
        status=status,
        headers=headers,
        detail="ok" if not problems else "; ".join(problems),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CORS behavior for Qwen-TTS API")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--request-method", default="POST")
    parser.add_argument("--request-headers", default="content-type")
    parser.add_argument("--machine-name", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        _check_preflight(args.api_base_url, args.origin, args.request_method, args.request_headers),
        _check_post_error_path(args.api_base_url, args.origin),
    ]
    passed = all(check.passed for check in checks)
    result = {
        "machine_name": args.machine_name,
        "run_tag": args.run_tag,
        "api_base_url": args.api_base_url,
        "origin": args.origin,
        "request_method": args.request_method,
        "request_headers": args.request_headers,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": [asdict(check) for check in checks],
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Qwen-TTS API CORS Regression",
        "",
        f"- machine: `{args.machine_name}`",
        f"- run tag: `{args.run_tag}`",
        f"- api: `{args.api_base_url}`",
        f"- origin: `{args.origin}`",
        f"- passed: `{passed}`",
        "",
        "## Checks",
        "",
    ]
    for check in checks:
        report_lines.append(f"- `{check.name}`: passed=`{check.passed}` status=`{check.status}` detail=`{check.detail}`")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if not passed:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"passed": True, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
