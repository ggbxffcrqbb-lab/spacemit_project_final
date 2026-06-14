from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from app.core.config import load_app_config
from app.core.logging_utils import setup_logging
from app.voice.service import ResidentVoiceService


DEFAULT_PROMPTS = [
    "\u8bf7\u7528\u4e00\u53e5\u7b80\u77ed\u4e2d\u6587\u4ecb\u7ecd\u4f60\u81ea\u5df1\u3002",
    "\u4eca\u5929\u6709\u70b9\u70ed\uff0c\u7ed9\u6211\u4e00\u4e2a\u7b80\u77ed\u51fa\u95e8\u5efa\u8bae\u3002",
    "\u628a\u201c\u73b0\u5728\u5df2\u7ecf\u5207\u5230\u6781\u901f\u6a21\u5f0f\u201d\u6539\u5199\u6210\u66f4\u81ea\u7136\u7684\u4e00\u53e5\u4e2d\u6587\u3002",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Compare default and fast voice modes")
    parser.add_argument(
        "--default-config",
        default="configs/voice.yaml",
        help="Config path for the default mode",
    )
    parser.add_argument(
        "--fast-config",
        default="configs/voice_fast.yaml",
        help="Config path for the fast mode",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path",
    )
    return parser


def run_mode(label: str, config_path: str | Path, prompts: list[str]) -> dict:
    config = load_app_config(config_path)
    setup_logging(
        log_dir=config.logging.dir,
        level=config.logging.level,
        runtime_file=config.logging.runtime_file,
    )

    service = ResidentVoiceService(config)
    try:
        service.start_workers()
        service.warmup()

        results = []
        for prompt in prompts:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                result = service.process_text_turn(prompt)
            results.append(
                {
                    "prompt": prompt,
                    "reply": result.reply_text,
                    "ttft_ms": result.first_chunk_ms,
                    "first_tts_enqueue_ms": result.first_tts_enqueue_ms,
                    "total_ms": result.total_ms,
                    "output_chars": result.output_chars,
                }
            )

        avg_ttft = round(sum(item["ttft_ms"] for item in results) / len(results), 2)
        avg_total = round(sum(item["total_ms"] for item in results) / len(results), 2)
        avg_chars = round(sum(item["output_chars"] for item in results) / len(results), 2)

        return {
            "mode": label,
            "model": config.voice.llm.model,
            "config_path": str(config.config_path),
            "prompts": results,
            "summary": {
                "avg_ttft_ms": avg_ttft,
                "avg_total_ms": avg_total,
                "avg_output_chars": avg_chars,
            },
        }
    finally:
        service.shutdown()


def main():
    parser = build_parser()
    args = parser.parse_args()

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "started_at": started_at,
        "modes": [
            run_mode("default", args.default_config, DEFAULT_PROMPTS),
            run_mode("fast", args.fast_config, DEFAULT_PROMPTS),
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
