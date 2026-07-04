from __future__ import annotations

import threading
from collections import deque
from copy import deepcopy
from typing import Any

from app.core.event_bus import AppEvent


PROJECT_LABELS = {
    "corrosion": "腐蚀疑似",
    "crevice_corrosion": "缝隙腐蚀",
    "pitting_corrosion": "点蚀",
    "uniform_corrosion": "均匀腐蚀",
    "rust_like_corrosion": "锈蚀疑似",
    "coating_flaking_or_delamination": "涂层剥落疑似",
    "chalking_or_powdering": "粉化疑似",
    "cui_risk_visual_hint": "保温层下腐蚀风险提示",
}


class MultimodalSessionState:
    def __init__(self, project_name: str, llm_model: str, history_limit: int = 8):
        self.project_name = project_name
        self.llm_model = llm_model
        self._lock = threading.RLock()
        self._events = deque(maxlen=max(4, history_limit))
        self._state: dict[str, Any] = {
            "project_name": project_name,
            "llm_model": llm_model,
            "mode": "启动中",
            "voice_stage": "booting",
            "voice_headline": "等待运行时启动",
            "voice_detail": "",
            "workers_started": False,
            "camera_backend": "",
            "recognizer_backend": "",
            "rag_document_count": 0,
            "rag_chunk_count": 0,
            "latest_user_text": "",
            "latest_reply_text": "",
            "latest_citations": [],
            "latest_voice_metrics": {},
            "latest_visual_candidates": [],
            "latest_visual_metrics": {},
            "latest_visual_notes": [],
            "latest_visual_summary": "等待视觉链路稳定输出",
            "latest_visual_context": "",
            "frame_index": 0,
            "source_frame_index": 0,
            "captured_frames": 0,
            "last_error": "",
        }

    def consume_event(self, event: AppEvent) -> None:
        with self._lock:
            self._events.appendleft(
                {
                    "stamp": event.stamp,
                    "kind": event.kind,
                    "summary": event.summary,
                }
            )
            if event.kind == "system.health":
                self._apply_health(event.payload)
            elif event.kind == "system.mode":
                self._state["mode"] = str(event.payload.get("mode") or event.summary)
            elif event.kind == "voice.status":
                self._apply_voice_status(event.payload)
            elif event.kind == "voice.turn":
                self._apply_voice_turn(event.payload)
            elif event.kind == "vision.analysis":
                self._apply_vision_analysis(event.payload)
            elif event.kind == "error":
                self._state["last_error"] = event.summary

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **deepcopy(self._state),
                "event_log": list(self._events),
            }

    def build_visual_context(self, max_candidates: int = 2) -> str:
        with self._lock:
            context = str(self._state.get("latest_visual_context", "")).strip()
            if context:
                return context

            candidates = list(self._state.get("latest_visual_candidates", []))[:max_candidates]
            if not candidates:
                return ""
            lines = [f"当前画面稳定候选 {len(candidates)} 个。"]
            for index, candidate in enumerate(candidates, start=1):
                label = _humanize_label(str(candidate.get("label", "")))
                score = float(candidate.get("score", 0.0))
                summary = str(candidate.get("summary", "")).strip()
                line = f"目标{index}：{label}，置信度 {score:.2f}。"
                if summary:
                    line += f"{summary}"
                lines.append(line)
            return "\n".join(lines)

    def build_assistant_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._state.get("mode", ""),
                "voice_stage": self._state.get("voice_stage", ""),
                "voice_headline": self._state.get("voice_headline", ""),
                "latest_user_text": self._state.get("latest_user_text", ""),
                "latest_reply_text": self._state.get("latest_reply_text", ""),
                "latest_citations": list(self._state.get("latest_citations", [])),
            }

    def _apply_health(self, payload: dict[str, Any]) -> None:
        rag_status = payload.get("rag_status") or {}
        self._state["workers_started"] = bool(payload.get("workers_started", False))
        self._state["rag_document_count"] = int(rag_status.get("document_count", 0) or 0)
        self._state["rag_chunk_count"] = int(rag_status.get("chunk_count", 0) or 0)

    def _apply_voice_status(self, payload: dict[str, Any]) -> None:
        if "stage" in payload:
            self._state["voice_stage"] = str(payload.get("stage", ""))
        if "headline" in payload:
            self._state["voice_headline"] = str(payload.get("headline", ""))
        if "detail" in payload:
            self._state["voice_detail"] = str(payload.get("detail", ""))
        latest_metrics = payload.get("latest_metrics")
        if latest_metrics is not None:
            self._state["latest_voice_metrics"] = dict(latest_metrics)

    def _apply_voice_turn(self, payload: dict[str, Any]) -> None:
        self._state["latest_user_text"] = str(payload.get("user_text", "")).strip()
        self._state["latest_reply_text"] = str(payload.get("reply_text", "")).strip()
        self._state["latest_citations"] = list(payload.get("citations", []))
        self._state["latest_voice_metrics"] = dict(payload.get("metrics", {}))
        self._state["voice_stage"] = "ready"
        self._state["voice_headline"] = "最近一轮问答已完成"
        self._state["voice_detail"] = "语音、RAG 与播报链路已完成一次闭环"

    def _apply_vision_analysis(self, payload: dict[str, Any]) -> None:
        result_dict = payload.get("result_dict") or payload.get("analysis") or {}
        candidates = list(result_dict.get("candidates", []))
        metrics = dict(result_dict.get("metrics", {}))
        notes = list(result_dict.get("notes", []))

        self._state["camera_backend"] = str(payload.get("camera_backend", self._state["camera_backend"]))
        self._state["recognizer_backend"] = str(
            payload.get("recognizer_backend", result_dict.get("recognizer_backend", self._state["recognizer_backend"]))
        )
        self._state["frame_index"] = int(payload.get("frame_index", self._state["frame_index"]) or 0)
        self._state["source_frame_index"] = int(
            payload.get("source_frame_index", self._state["source_frame_index"]) or 0
        )
        self._state["captured_frames"] = int(payload.get("captured_frames", self._state["captured_frames"]) or 0)
        self._state["latest_visual_candidates"] = candidates
        self._state["latest_visual_metrics"] = metrics
        self._state["latest_visual_notes"] = notes
        self._state["latest_visual_summary"] = _build_visual_summary(candidates, metrics)
        self._state["latest_visual_context"] = _build_visual_context(candidates, metrics)


def _humanize_label(label: str) -> str:
    return PROJECT_LABELS.get(label, label.replace("_", " ").strip() or "未命名目标")


def _build_visual_summary(candidates: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    if not candidates:
        seg_ms = metrics.get("seg_infer_ms")
        if seg_ms is None:
            return "当前没有稳定高置信候选，视觉链路仍在持续刷新。"
        return f"当前没有稳定高置信候选，最近一轮分割耗时约 {seg_ms} ms。"

    top = candidates[0]
    label = _humanize_label(str(top.get("label", "")))
    score = float(top.get("score", 0.0))
    evidence = dict(top.get("evidence", {}))
    seg_score = evidence.get("segmentation_score")
    cls_score = evidence.get("classification_score")
    parts = [f"当前主候选为 {label}，置信度 {score:.2f}。"]
    if seg_score not in {None, ""}:
        parts.append(f"分割得分 {float(seg_score):.2f}。")
    if cls_score not in {None, ""} and float(cls_score) > 0.0:
        parts.append(f"分类得分 {float(cls_score):.2f}。")
    seg_ms = metrics.get("seg_infer_ms")
    if seg_ms is not None:
        parts.append(f"最近一轮分割耗时约 {seg_ms} ms。")
    return " ".join(parts)


def _build_visual_context(candidates: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    if not candidates:
        seg_ms = metrics.get("seg_infer_ms")
        if seg_ms is None:
            return ""
        return f"当前画面暂未稳定识别出高置信缺陷目标，最近一轮分割耗时约 {seg_ms} ms。"

    lines = [f"当前画面稳定候选 {min(len(candidates), 2)} 个。"]
    for index, candidate in enumerate(candidates[:2], start=1):
        label = _humanize_label(str(candidate.get("label", "")))
        score = float(candidate.get("score", 0.0))
        evidence = dict(candidate.get("evidence", {}))
        seg_score = evidence.get("segmentation_score")
        cls_score = evidence.get("classification_score")
        detail = f"目标{index}：{label}，置信度 {score:.2f}"
        if seg_score not in {None, ""}:
            detail += f"，分割得分 {float(seg_score):.2f}"
        if cls_score not in {None, ""} and float(cls_score) > 0.0:
            detail += f"，分类得分 {float(cls_score):.2f}"
        detail += "。"
        lines.append(detail)
    seg_ms = metrics.get("seg_infer_ms")
    if seg_ms is not None:
        lines.append(f"最近一轮视觉分割耗时约 {seg_ms} ms。")
    return "\n".join(lines)
