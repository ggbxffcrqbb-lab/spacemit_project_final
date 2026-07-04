from __future__ import annotations

import shutil
import sys
import textwrap
from typing import Any
from typing import TextIO


class TerminalDashboard:
    def __init__(
        self,
        title: str = "Muse Pi Pro Multimodal Demo",
        output_stream: TextIO | None = None,
    ):
        self.title = title
        self.output_stream = output_stream or sys.stdout

    def redraw(self, snapshot: dict[str, Any]) -> None:
        self.output_stream.write("\x1b[2J\x1b[H")
        self.output_stream.write(self.render(snapshot))
        self.output_stream.flush()

    def render(self, snapshot: dict[str, Any]) -> str:
        width = max(96, min(140, shutil.get_terminal_size((120, 40)).columns))
        lines: list[str] = []
        lines.extend(
            [
                "=" * width,
                self._fit(self.title, width),
                "=" * width,
                self._fit(
                    "MODE: {mode} | MIC: {voice_stage} | CAM: {camera_backend} | RAG: {docs} docs / {chunks} chunks".format(
                        mode=snapshot.get("mode", "未知"),
                        voice_stage=snapshot.get("voice_stage", "unknown"),
                        camera_backend=snapshot.get("camera_backend", "pending"),
                        docs=snapshot.get("rag_document_count", 0),
                        chunks=snapshot.get("rag_chunk_count", 0),
                    ),
                    width,
                ),
                "-" * width,
                self._section("Voice", width),
                self._fit(f"Status: {snapshot.get('voice_headline', '')}", width),
                self._fit(f"Detail: {snapshot.get('voice_detail', '')}", width),
                self._fit(f"User: {snapshot.get('latest_user_text', '') or '暂无'}", width),
                self._fit(f"Reply: {snapshot.get('latest_reply_text', '') or '暂无'}", width),
                self._fit(
                    "Citations: {value}".format(
                        value=" | ".join(snapshot.get("latest_citations", [])) or "暂无"
                    ),
                    width,
                ),
                self._fit(
                    "Voice Metrics: {value}".format(
                        value=self._format_metrics(snapshot.get("latest_voice_metrics", {}))
                    ),
                    width,
                ),
                "-" * width,
                self._section("Vision", width),
                self._fit(f"Summary: {snapshot.get('latest_visual_summary', '')}", width),
                self._fit(
                    "Candidates: {value}".format(
                        value=self._format_candidates(snapshot.get("latest_visual_candidates", []))
                    ),
                    width,
                ),
                self._fit(
                    "Vision Metrics: {value}".format(
                        value=self._format_metrics(snapshot.get("latest_visual_metrics", {}))
                    ),
                    width,
                ),
                self._fit(
                    "Frames: display={display} | source={source} | captured={captured}".format(
                        display=snapshot.get("frame_index", 0),
                        source=snapshot.get("source_frame_index", 0),
                        captured=snapshot.get("captured_frames", 0),
                    ),
                    width,
                ),
                "-" * width,
                self._section("Event Log", width),
            ]
        )

        events = snapshot.get("event_log", []) or []
        if not events:
            lines.append(self._fit("暂无事件。", width))
        else:
            for item in events[:8]:
                lines.append(
                    self._fit(
                        "[{stamp}] {kind}: {summary}".format(
                            stamp=item.get("stamp", "--:--:--"),
                            kind=item.get("kind", "event"),
                            summary=item.get("summary", ""),
                        ),
                        width,
                    )
                )

        last_error = str(snapshot.get("last_error", "")).strip()
        if last_error:
            lines.extend(
                [
                    "-" * width,
                    self._section("Last Error", width),
                    self._fit(last_error, width),
                ]
            )

        lines.extend(
            [
                "-" * width,
                self._fit(
                    "Commands: [Enter/A] audio  [T] text  [V] summary  [S] snapshot  [Q] quit",
                    width,
                ),
                "=" * width,
            ]
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _section(label: str, width: int) -> str:
        return f"[{label}]".ljust(width)

    @staticmethod
    def _fit(text: str, width: int) -> str:
        wrapped = textwrap.wrap(text, width=width, replace_whitespace=False) or [""]
        return "\n".join(line.ljust(width) for line in wrapped)

    @staticmethod
    def _format_metrics(metrics: dict[str, Any]) -> str:
        if not metrics:
            return "暂无"
        parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
            if len(parts) >= 6:
                break
        return " | ".join(parts)

    @staticmethod
    def _format_candidates(candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return "暂无稳定目标"
        parts = []
        for candidate in candidates[:3]:
            label = str(candidate.get("label", "unknown"))
            score = float(candidate.get("score", 0.0))
            parts.append(f"{label}:{score:.2f}")
        return " | ".join(parts)
