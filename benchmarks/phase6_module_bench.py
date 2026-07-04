#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import yaml

from app.core.config import load_app_config
from app.voice.asr_runtime import AsrModel
from app.voice.llm_runtime.deepseek_openai import LlmModel
from app.vision.recognizer import build_defect_recognizer


DEFAULT_LLM_PROMPTS = [
    "请用一句中文说出钢管表面生锈后的第一步处理建议。",
    "如果现场看到钢管表面有锈蚀，给我一个保守、简短的建议。",
]


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _current_affinity() -> str:
    if not hasattr(os, "sched_getaffinity"):
        return ""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except OSError:
        return ""
    if not cpus:
        return ""
    ranges: list[str] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _summarize_numeric(rows: list[dict], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
    }


def _append_jsonl(path_text: str, payload: dict) -> None:
    if not path_text:
        return
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _audio_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() / float(wav_file.getframerate())
    except Exception:
        return None


def _benchmark_asr(args) -> dict:
    config = load_app_config(args.config)
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"ASR audio not found: {audio_path}")

    init_begin = _now_ms()
    model = AsrModel(
        model_dir=str(config.voice.asr.model_dir),
        prefer_optimized_model=config.voice.asr.prefer_optimized_model,
        batch_size=config.voice.asr.batch_size,
        language=config.voice.asr.language,
        use_itn=config.voice.asr.use_itn,
        intra_op_num_threads=int(args.threads),
    )
    init_wall_ms = round(_now_ms() - init_begin, 3)

    rows = []
    for run_idx in range(1, args.repeat + 1):
        started_at = _now_ms()
        text = model(str(audio_path))
        wall_ms = round(_now_ms() - started_at, 3)
        rows.append(
            {
                "run": run_idx,
                "wall_ms": wall_ms,
                "text_chars": len(text),
                "text": text,
            }
        )

    payload = {
        "module": "asr",
        "tag": args.tag,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "process_affinity": _current_affinity(),
        "threads": int(args.threads),
        "audio_path": str(audio_path),
        "audio_seconds": _audio_duration_seconds(audio_path),
        "init_wall_ms": init_wall_ms,
        "runs": rows,
        "summary": {
            "wall_ms": _summarize_numeric(rows, "wall_ms"),
            "text_chars": _summarize_numeric(rows, "text_chars"),
        },
    }
    return payload


def _list_ollama_processes() -> list[dict[str, str | int]]:
    try:
        result = subprocess.run(
            ["ps", "-u", "ollama", "-o", "pid=", "-o", "args="],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    rows: list[dict[str, str | int]] = []
    for line in (result.stdout or "").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split(None, 1)
        if not parts:
            continue
        pid = int(parts[0])
        cmdline = parts[1] if len(parts) > 1 else ""
        rows.append({"pid": pid, "cmdline": cmdline})
    return rows


def _benchmark_llm(args) -> dict:
    config = load_app_config(args.config)
    llm_cfg = config.voice.llm
    prompts = list(args.prompt) if args.prompt else list(DEFAULT_LLM_PROMPTS)

    if args.stop_model_before:
        subprocess.run(
            ["ollama", "stop", llm_cfg.model],
            check=False,
            capture_output=True,
            text=True,
        )
        time.sleep(1.0)

    model = LlmModel(
        model_path=llm_cfg.model,
        system_prompt=llm_cfg.system_prompt,
        max_chars=llm_cfg.max_chars,
        min_chars=llm_cfg.min_chars,
        stop_after_first_sentence=llm_cfg.stop_after_first_sentence,
        num_thread=int(args.threads),
    )

    warmup_wall_ms = None
    if args.warmup:
        warmup_begin = _now_ms()
        for _chunk in model.generate(llm_cfg.warmup_prompt):
            pass
        warmup_wall_ms = round(_now_ms() - warmup_begin, 3)

    process_rows: list[dict] = []
    for repeat_idx in range(1, args.repeat + 1):
        for prompt_idx, prompt in enumerate(prompts, start=1):
            turn_begin = _now_ms()
            first_chunk_at = None
            chunks: list[str] = []
            for chunk in model.generate(prompt):
                if first_chunk_at is None:
                    first_chunk_at = _now_ms()
                chunks.append(chunk)
            total_wall_ms = round(_now_ms() - turn_begin, 3)
            reply = "".join(chunks)
            process_rows.append(
                {
                    "repeat": repeat_idx,
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "reply": reply,
                    "output_chars": len(reply),
                    "ttft_ms": -1 if first_chunk_at is None else round(first_chunk_at - turn_begin, 3),
                    "total_ms": total_wall_ms,
                }
            )

    payload = {
        "module": "llm",
        "tag": args.tag,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "process_affinity": _current_affinity(),
        "threads": int(args.threads),
        "model": llm_cfg.model,
        "warmup": bool(args.warmup),
        "warmup_wall_ms": warmup_wall_ms,
        "ollama_processes": _list_ollama_processes(),
        "runs": process_rows,
        "summary": {
            "ttft_ms": _summarize_numeric(process_rows, "ttft_ms"),
            "total_ms": _summarize_numeric(process_rows, "total_ms"),
            "output_chars": _summarize_numeric(process_rows, "output_chars"),
        },
    }
    return payload


def _write_temp_seg_config(base_config_path: Path, seg_threads: int) -> Path:
    payload = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    default_params = payload.setdefault("default_params", {})
    default_params["num_threads"] = int(seg_threads)

    fd, temp_path_text = tempfile.mkstemp(prefix="vision_seg_bench_", suffix=".yaml")
    os.close(fd)
    temp_path = Path(temp_path_text)
    temp_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return temp_path


def _benchmark_vision(args) -> dict:
    config = load_app_config(args.config)
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Vision image not found: {image_path}")

    base_seg_config = Path(str(config.vision.recognizer.spacemit_vision_config or "")).expanduser()
    if not base_seg_config.exists():
        raise FileNotFoundError(f"Vision seg config not found: {base_seg_config}")

    temp_seg_config = _write_temp_seg_config(base_seg_config, int(args.seg_threads))
    try:
        config.vision.recognizer.spacemit_vision_config = temp_seg_config
        config.vision.recognizer.options = dict(config.vision.recognizer.options or {})
        config.vision.recognizer.options["cls_cpu_threads"] = int(args.cls_threads)

        init_begin = _now_ms()
        recognizer = build_defect_recognizer(config.vision, config.vision.recognizer.backend)
        init_wall_ms = round(_now_ms() - init_begin, 3)

        rows = []
        for run_idx in range(1, args.repeat + 1):
            started_at = _now_ms()
            result = recognizer.analyze(image_path)
            wall_ms = round(_now_ms() - started_at, 3)
            rows.append(
                {
                    "run": run_idx,
                    "wall_ms": wall_ms,
                    "seg_infer_ms": float(result.metrics.get("seg_infer_ms", -1.0)),
                    "candidate_count": len(result.candidates),
                    "top_label": result.candidates[0].label if result.candidates else "",
                    "top_score": float(result.candidates[0].score) if result.candidates else 0.0,
                }
            )
    finally:
        try:
            temp_seg_config.unlink(missing_ok=True)
        except OSError:
            pass

    payload = {
        "module": "vision",
        "tag": args.tag,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "process_affinity": _current_affinity(),
        "backend": config.vision.recognizer.backend,
        "image_path": str(image_path),
        "seg_threads": int(args.seg_threads),
        "cls_threads": int(args.cls_threads),
        "init_wall_ms": init_wall_ms,
        "runs": rows,
        "summary": {
            "wall_ms": _summarize_numeric(rows, "wall_ms"),
            "seg_infer_ms": _summarize_numeric(rows, "seg_infer_ms"),
            "candidate_count": _summarize_numeric(rows, "candidate_count"),
        },
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Board-side Phase6 module benchmark")
    parser.add_argument(
        "--config",
        default="configs/multimodal_demo.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repeated timed runs",
    )
    parser.add_argument(
        "--jsonl",
        default="",
        help="Append JSON result to a jsonl file",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag for grouping results",
    )

    subparsers = parser.add_subparsers(dest="module", required=True)

    asr_parser = subparsers.add_parser("asr", help="Benchmark ASR")
    asr_parser.add_argument("--threads", type=int, required=True, help="ASR intra-op thread count")
    asr_parser.add_argument("--audio", required=True, help="Input wav path")

    llm_parser = subparsers.add_parser("llm", help="Benchmark LLM")
    llm_parser.add_argument("--threads", type=int, required=True, help="LLM num_thread")
    llm_parser.add_argument("--prompt", action="append", default=[], help="Prompt text; repeatable")
    llm_parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm model once before timed runs",
    )
    llm_parser.add_argument(
        "--stop-model-before",
        action="store_true",
        help="Call `ollama stop <model>` before warmup/benchmark",
    )

    vision_parser = subparsers.add_parser("vision", help="Benchmark vision recognizer")
    vision_parser.add_argument("--seg-threads", type=int, required=True, help="Segmentation num_threads")
    vision_parser.add_argument("--cls-threads", type=int, required=True, help="Classification cpu threads")
    vision_parser.add_argument("--image", required=True, help="Input image path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.module == "asr":
        payload = _benchmark_asr(args)
    elif args.module == "llm":
        payload = _benchmark_llm(args)
    elif args.module == "vision":
        payload = _benchmark_vision(args)
    else:
        raise ValueError(f"Unsupported module: {args.module}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    _append_jsonl(args.jsonl, payload)


if __name__ == "__main__":
    main()
