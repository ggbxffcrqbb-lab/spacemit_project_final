#!/usr/bin/env python3
"""
Board-side benchmark for spacemit-com/model-zoo-tts.

Goals:
1. Measure engine init, warmup and blocking synth cost.
2. Compare preset / provider / num_threads / speech_rate.
3. Optionally measure save-to-file overhead.

This script is intentionally self-contained and can run even when
`spacemit_tts` is not installed system-wide.
"""

from __future__ import annotations

import argparse
import json
import os
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
    "mix_short": "今天学Python。",
    "mix_medium": "今天我们在 Muse Pi Pro 上评估 Matcha TTS 的 provider, thread count and speech rate。",
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
    parser = argparse.ArgumentParser(description="Benchmark model-zoo-tts on board")
    parser.add_argument("--repo-root", required=True, help="Path to model-zoo-tts repo on board")
    parser.add_argument("--preset", default="matcha_zh", help="Preset name, e.g. matcha_zh")
    parser.add_argument("--provider", default="cpu", help="Provider: cpu/auto/spacemit")
    parser.add_argument("--threads", type=int, default=3, help="num_threads")
    parser.add_argument("--speech-rate", type=float, default=1.0, help="speech rate")
    parser.add_argument("--repeat", type=int, default=3, help="repetitions")
    parser.add_argument("--text-key", default="zh_medium", choices=sorted(TEXT_LIBRARY.keys()))
    parser.add_argument("--text", default="", help="Custom text overrides text-key")
    parser.add_argument("--warmup", action="store_true", help="Warm up before timed runs")
    parser.add_argument("--measure-save", action="store_true", help="Measure save-to-file overhead")
    parser.add_argument("--output-dir", default="", help="Directory for temporary wav outputs")
    parser.add_argument("--jsonl", default="", help="Append each run result to jsonl file")
    parser.add_argument("--tag", default="", help="Optional tag to mark this benchmark batch")
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
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="tts_bench_out_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    spacemit_tts = ensure_spacemit_tts(repo_root)

    init_begin = now_ms()
    engine, config = configure_engine(spacemit_tts, args)
    init_end = now_ms()

    if not engine.is_initialized:
        raise RuntimeError("Engine initialization failed")

    warmup_row = None
    if args.warmup:
        warmup_begin = now_ms()
        warmup_result = engine.synthesize("你好。")
        warmup_end = now_ms()
        warmup_row = {
            "wall_ms": round(warmup_end - warmup_begin, 3),
            "duration_ms": warmup_result.duration_ms,
            "processing_time_ms": warmup_result.processing_time_ms,
            "rtf": round(warmup_result.rtf, 6),
            "sample_rate": warmup_result.sample_rate,
        }

    rows = []
    for run_idx in range(1, args.repeat + 1):
        synth_begin = now_ms()
        result = engine.synthesize(text)
        synth_end = now_ms()
        if not result.is_success:
            raise RuntimeError(f"synthesis failed on run {run_idx}: {result.message}")

        save_wall_ms = None
        save_path = None
        if args.measure_save:
            save_path = output_dir / f"{args.preset}_{args.provider}_t{args.threads}_r{run_idx}.wav"
            save_begin = now_ms()
            ok = result.save(save_path)
            save_end = now_ms()
            if not ok:
                raise RuntimeError(f"save_to_file failed on run {run_idx}: {save_path}")
            save_wall_ms = round(save_end - save_begin, 3)

        rows.append(
            {
                "run": run_idx,
                "wall_ms": round(synth_end - synth_begin, 3),
                "duration_ms": result.duration_ms,
                "processing_time_ms": result.processing_time_ms,
                "rtf": round(result.rtf, 6),
                "sample_rate": result.sample_rate,
                "audio_samples": len(result.audio_int16),
                "save_wall_ms": save_wall_ms,
                "save_path": str(save_path) if save_path else "",
            }
        )

    summary = {
        "tag": args.tag,
        "repo_root": str(repo_root),
        "preset": args.preset,
        "provider": args.provider,
        "threads": args.threads,
        "speech_rate": args.speech_rate,
        "text_key": args.text_key,
        "text_chars": len(text),
        "text": text,
        "engine_name": engine.engine_name,
        "engine_sample_rate": engine.sample_rate,
        "init_wall_ms": round(init_end - init_begin, 3),
        "warmup": warmup_row,
        "runs": rows,
        "summary": {
            "wall_ms": summarize_numeric(rows, "wall_ms"),
            "duration_ms": summarize_numeric(rows, "duration_ms"),
            "processing_time_ms": summarize_numeric(rows, "processing_time_ms"),
            "rtf": summarize_numeric(rows, "rtf"),
        },
    }

    if args.measure_save:
        save_values = [row["save_wall_ms"] for row in rows if row["save_wall_ms"] is not None]
        if save_values:
            summary["summary"]["save_wall_ms"] = {
                "min": round(min(save_values), 3),
                "max": round(max(save_values), 3),
                "mean": round(statistics.mean(save_values), 3),
                "median": round(statistics.median(save_values), 3),
            }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
