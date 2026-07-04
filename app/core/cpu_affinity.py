from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Iterable


def parse_cpuset(spec: str | Iterable[int] | None) -> set[int] | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return None
        cpus: set[int] = set()
        for part in text.split(","):
            chunk = part.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start_text, end_text = chunk.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    start, end = end, start
                cpus.update(range(start, end + 1))
            else:
                cpus.add(int(chunk))
        return cpus or None

    cpus = {int(item) for item in spec}
    return cpus or None


def format_cpuset(spec: str | Iterable[int] | None) -> str:
    cpus = sorted(parse_cpuset(spec) or [])
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


def current_thread_affinity() -> set[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return set(os.sched_getaffinity(0))
    except OSError:
        return None


def bind_current_thread(
    cpuset: str | Iterable[int] | None,
    *,
    logger: logging.Logger | None = None,
    label: str = "",
) -> set[int] | None:
    mask = parse_cpuset(cpuset)
    if not mask or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        os.sched_setaffinity(0, mask)
        actual = set(os.sched_getaffinity(0))
    except OSError as exc:
        if logger is not None:
            logger.warning("Failed to bind current thread for %s: %s", label or "thread", exc)
        return None

    if logger is not None:
        logger.info(
            "CPU affinity applied for %s: requested=%s actual=%s",
            label or "thread",
            format_cpuset(mask),
            format_cpuset(actual),
        )
    return actual


@contextlib.contextmanager
def affinity_scope(
    cpuset: str | Iterable[int] | None,
    *,
    logger: logging.Logger | None = None,
    label: str = "",
):
    mask = parse_cpuset(cpuset)
    if not mask or not hasattr(os, "sched_setaffinity"):
        yield
        return

    original = current_thread_affinity()
    bind_current_thread(mask, logger=logger, label=label)
    try:
        yield
    finally:
        if original is not None:
            try:
                os.sched_setaffinity(0, original)
            except OSError as exc:
                if logger is not None:
                    logger.warning(
                        "Failed to restore CPU affinity for %s: %s",
                        label or "thread",
                        exc,
                    )


def bind_process(
    pid: int,
    cpuset: str | Iterable[int] | None,
    *,
    logger: logging.Logger | None = None,
    label: str = "",
) -> set[int] | None:
    mask = parse_cpuset(cpuset)
    if not mask or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        os.sched_setaffinity(int(pid), mask)
        actual = set(os.sched_getaffinity(int(pid)))
        return actual
    except OSError as exc:
        if logger is not None:
            logger.warning("Failed to bind process %s for %s: %s", pid, label or "process", exc)
        return None


def find_processes_by_cmdline(substrings: Iterable[str]) -> list[tuple[int, str]]:
    required = [str(item).strip() for item in substrings if str(item).strip()]
    if not required:
        return []

    results: list[tuple[int, str]] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore").strip()
        except OSError:
            continue
        if not cmdline:
            continue
        if all(fragment in cmdline for fragment in required):
            results.append((int(entry.name), cmdline))
    return results
