from __future__ import annotations

import argparse
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

DEFAULT_MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS models from ModelScope.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-root", default="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        local_dir = output_root / model_id.split("/")[-1]
        print(f"[download] {model_id} -> {local_dir}")
        snapshot_download(model_id, local_dir=str(local_dir))


if __name__ == "__main__":
    main()
