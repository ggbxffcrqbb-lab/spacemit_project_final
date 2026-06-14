#!/usr/bin/env python3
"""
Resident-engine benchmark for spacemit-com/model-zoo-tts on Muse Pi Pro.

Measures:
1. One-time engine initialization cost.
2. Sequential synthesis cost when the same engine stays alive.
3. Optional gap/warmup behavior across repeated requests.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path


TEXT_LIBRARY = {
    "zh_short": "你好，这是短句测试。",
    "zh_medium": "你好，这是 Muse Pi Pro 上的语音合成性能测试，我们需要关注首包时延和整体推理效率。",
    "zh_long": (
        "今天我们对板端文本转语音组件进行系统化评估，"
        "目标不是只看单次推理速度，而是同时观察初始化开销、"
        "热态处理时间、输出音频时长，以及不同参数配置对整体体感延迟的影响。"
    ),
    "mix_short": "今天学 Python。",
    "mix_medium": "今天我们在 Muse Pi Pro 上评估 Matcha TTS 的 provider、thread count 和 speech rate。",
}


def ensure_spacemit_tts(repo_root: Path):
    source_pkg = repo_root / "python" / "spacemit_tts"
    native_dir = repo_root / "build" / "python"
    so_candidates = sorted(native_dir.glob("_spacemit_tts*.so"))
    if source_pkg.exists() and so_candidates:
        temp_root = Path(tempfile.mkdtemp(prefix="spacemit_tts_pkg_"))
        temp_pkg = temp_root / "spacemit_tts"
        shutil.copytree(source_pkg, temp_pkg)
        shutil.copy2(so_candidates[0], temp_pkg / so_candidates[0].name)
        sys.path.insert(0, str(temp_root))
        import spacemit_tts  # type: ignore

        return spacemit_tts

    try:
        import spacemit_tts  # type: ignore

        return spacemit_tts
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import spacemit_tts and repo-local package assembly failed. "
            f"source_pkg={source_pkg} so_candidates={len(so_candidates)}"
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark resident TTS engine on board")
    parser.add_argument("--repo-root", required=True, help="Path to model-zoo-tts repo on board")
    parser.add_argument("--preset", default="matcha_zh", help="Preset name, e.g. matcha_zh")
    parser.add_argument("--provider", default="cpu", help="Provider: cpu/auto/spacemit")
    parser.add_argument("--threads", type=int, default=3, help="num_threads")
    parser.add_argument("--speech-rate", type=float, default=1.0, help="speech rate")
    parser.add_argument("--requests", type=int, default=5, help="Sequential synth requests on one engine")
    parser.add_argument("--text-key", default="zh_short", choices=sorted(TEXT_LIBRARY.keys()))
    parser.add_argument("--text", default="", help="Custom text overrides text-key")
    parser.add_argument("--warmup", action="store_true", help="Enable engine warmup during initialization")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between requests to simulate gaps")
    parser.add_argument("--jsonl", default="", help="Append result to jsonl file")
    parser.add_argument("--tag", default="", help="Optional tag for this benchmark batch")
    return parser.parse_args()


def configure_engine(spacemit_tts, args):
    config = spacemit_tts.Config.preset(args.preset)
    config.provider = args.provider
    config.speech_rate = args.speech_rate
    config.num_threads = args.threads
    config._config.enable_warmup = bool(args.warmup)
    return spacemit_tts.Engine(config), config


def pick_text(args):
    return args.text if args.text else TEXT_LIBRARY[args.text_key]


def now_ms():
    return time.perf_counter() * 1000.0


def summarize_numeric(rows, key):
    values = [row[key] for row in rows]
    return {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
    }


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    text = pick_text(args)
    spacemit_tts = ensure_spacemit_tts(repo_root)

    init_begin = now_ms()
    engine, _config = configure_engine(spacemit_tts, args)
    init_end = now_ms()

    if not engine.is_initialized:
        raise RuntimeError("Engine initialization failed")

    rows = []
    for request_idx in range(1, args.requests + 1):
        synth_begin = now_ms()
        result = engine.synthesize(text)
        synth_end = now_ms()
        if not result.is_success:
            raise RuntimeError(f"synthesis failed on request {request_idx}: {result.message}")

        row = {
            "request": request_idx,
            "wall_ms": round(synth_end - synth_begin, 3),
            "duration_ms": result.duration_ms,
            "processing_time_ms": result.processing_time_ms,
            "rtf": round(result.rtf, 6),
            "sample_rate": result.sample_rate,
            "audio_samples": len(result.audio_int16),
        }
        rows.append(row)

        if args.sleep_ms > 0 and request_idx < args.requests:
            time.sleep(args.sleep_ms / 1000.0)

    first_row = rows[0]
    tail_rows = rows[1:] if len(rows) > 1 else rows

    summary = {
        "tag": args.tag,
        "repo_root": str(repo_root),
        "preset": args.preset,
        "provider": args.provider,
        "threads": args.threads,
        "speech_rate": args.speech_rate,
        "warmup": bool(args.warmup),
        "requests": args.requests,
        "sleep_ms": args.sleep_ms,
        "text_key": args.text_key,
        "text_chars": len(text),
        "text": text,
        "engine_name": engine.engine_name,
        "engine_sample_rate": engine.sample_rate,
        "init_wall_ms": round(init_end - init_begin, 3),
        "runs": rows,
        "summary": {
            "all_wall_ms": summarize_numeric(rows, "wall_ms"),
            "all_processing_time_ms": summarize_numeric(rows, "processing_time_ms"),
            "all_rtf": summarize_numeric(rows, "rtf"),
            "first_request": first_row,
            "steady_state_wall_ms": summarize_numeric(tail_rows, "wall_ms"),
            "steady_state_processing_time_ms": summarize_numeric(tail_rows, "processing_time_ms"),
            "steady_state_rtf": summarize_numeric(tail_rows, "rtf"),
            "cold_to_steady_wall_delta_ms": round(first_row["wall_ms"] - statistics.mean([r["wall_ms"] for r in tail_rows]), 3),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
