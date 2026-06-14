#!/usr/bin/env python3
"""
Run matrix benchmarks for model-zoo-tts on board.

This is a thin orchestration layer on top of bench_tts_matrix.py.
It keeps going on failures and writes a separate failure log.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_csv(value: str, cast=str):
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return [cast(item) for item in parts]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep preset/provider/thread/rate matrix")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bench-script", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--presets", default="matcha_zh")
    parser.add_argument("--providers", default="cpu")
    parser.add_argument("--threads", default="1,2,3,4")
    parser.add_argument("--speech-rates", default="1.0")
    parser.add_argument("--text-keys", default="zh_medium")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--measure-save", action="store_true")
    parser.add_argument("--tag", default="sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    success_jsonl = log_dir / f"{args.tag}_success.jsonl"
    failure_jsonl = log_dir / f"{args.tag}_failure.jsonl"

    presets = parse_csv(args.presets, str)
    providers = parse_csv(args.providers, str)
    threads = parse_csv(args.threads, int)
    speech_rates = parse_csv(args.speech_rates, float)
    text_keys = parse_csv(args.text_keys, str)

    total = 0
    ok = 0
    failed = 0

    for preset, provider, thread_count, speech_rate, text_key in itertools.product(
        presets, providers, threads, speech_rates, text_keys
    ):
        total += 1
        tag = f"{args.tag}:{preset}:{provider}:t{thread_count}:r{speech_rate}:{text_key}"
        cmd = [
            args.python,
            args.bench_script,
            "--repo-root",
            args.repo_root,
            "--preset",
            preset,
            "--provider",
            provider,
            "--threads",
            str(thread_count),
            "--speech-rate",
            str(speech_rate),
            "--text-key",
            text_key,
            "--repeat",
            str(args.repeat),
            "--jsonl",
            str(success_jsonl),
            "--tag",
            tag,
        ]
        if args.warmup:
            cmd.append("--warmup")
        if args.measure_save:
            cmd.append("--measure-save")

        print(f"[RUN {total}] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            ok += 1
            print(proc.stdout, flush=True)
        else:
            failed += 1
            payload = {
                "tag": tag,
                "returncode": proc.returncode,
                "cmd": cmd,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            with failure_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"[FAIL] {tag}", flush=True)
            if proc.stdout:
                print(proc.stdout, flush=True)
            if proc.stderr:
                print(proc.stderr, flush=True)

    print(
        json.dumps(
            {
                "tag": args.tag,
                "total": total,
                "ok": ok,
                "failed": failed,
                "success_jsonl": str(success_jsonl),
                "failure_jsonl": str(failure_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
