from __future__ import annotations

import logging
import os
import select
import sys
import threading
from typing import Any
from typing import TextIO

from app.core.config import AppConfig
from app.core.cpu_affinity import bind_current_thread
from app.core.event_bus import EventBus
from app.core.session_state import MultimodalSessionState
from app.ui.terminal_dashboard import TerminalDashboard
from app.vision.service import VisionPipelineService
from app.voice.service import ResidentVoiceService, VoiceTurnResult


class MultimodalDemoController:
    _VISUAL_CONTEXT_STRONG_TERMS = (
        "画面",
        "屏幕",
        "图里",
        "图中",
        "图上",
        "镜头",
        "相机",
        "摄像头",
        "视频里",
        "视频中",
        "照片里",
        "照片中",
        "当前场景",
        "当前画面",
        "看见",
        "看到",
    )
    _VISUAL_CONTEXT_DEICTIC_TERMS = (
        "这种",
        "这个位置",
        "这个地方",
        "这里",
        "这块",
        "这处",
        "这一片",
        "上面这个",
        "下面这个",
        "左边这个",
        "右边这个",
        "里面这个",
    )
    _VISUAL_CONTEXT_OBJECT_TERMS = (
        "锈",
        "腐蚀",
        "返锈",
        "起泡",
        "粉化",
        "剥落",
        "裂纹",
        "缺陷",
        "区域",
        "部位",
        "目标",
        "位置",
        "焊缝",
        "法兰",
        "支架",
        "管道",
        "钢管",
        "储罐",
        "保温",
    )

    def __init__(
        self,
        config: AppConfig,
        *,
        camera_backend: str | None = None,
        recognizer_backend: str | None = None,
        interval_seconds: float = 0.0,
        max_frames: int = 0,
        display_competition: bool = True,
        performance_mode: bool = True,
    ):
        self.config = config
        self.logger = logging.getLogger("app.core.multimodal_controller")
        self.camera_backend = camera_backend or None
        self.recognizer_backend = recognizer_backend or None
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.max_frames = max(0, int(max_frames))
        self.display_competition = bool(display_competition)
        self.performance_mode = bool(performance_mode)

        self.stop_event = threading.Event()
        self.event_bus = EventBus()
        self.session_state = MultimodalSessionState(
            project_name=config.project_name,
            llm_model=config.voice.llm.model,
            history_limit=max(8, config.ui.history_limit),
        )
        self.event_bus.subscribe(self.session_state.consume_event)

        self._interactive_console = sys.stdin.isatty()
        self._console_stream, self._owns_console_stream = self._resolve_console_stream()
        self._dashboard_enabled = self._console_stream is not None
        self.dashboard = TerminalDashboard(
            title=f"{config.project_name} Phase 6 Multimodal Console",
            output_stream=self._console_stream,
        )

        self.voice_service = ResidentVoiceService(config, status_hook=self._on_voice_status)
        self.vision_service = VisionPipelineService(config)
        self._input_thread: threading.Thread | None = None

    def run(self) -> dict[str, Any]:
        self.event_bus.publish("system.mode", "Enter inspection mode", {"mode": "Inspection"})
        self.voice_service.start_workers()
        self.voice_service.warmup()
        self.event_bus.publish(
            "system.health",
            "Voice and knowledge base warmup complete",
            self.voice_service.build_health_report(),
        )

        if self._interactive_console:
            if not self._dashboard_enabled:
                self.event_bus.publish(
                    "system.info",
                    "Interactive input is available, but no TTY output stream was found.",
                    {},
                )
            self._input_thread = threading.Thread(
                target=self._control_loop,
                name="multimodal-demo-control",
                daemon=True,
            )
            self._input_thread.start()
        else:
            self.event_bus.publish(
                "system.info",
                "stdin is not interactive; terminal commands are disabled. Use ESC/Q on the fullscreen UI or Ctrl+C.",
                {},
            )

        try:
            result = self.vision_service.stream_camera(
                camera_backend=self.camera_backend,
                recognizer_backend=self.recognizer_backend,
                interval_seconds=self.interval_seconds,
                max_frames=self.max_frames,
                display_status_page=False,
                performance_mode=self.performance_mode,
                display_competition=self.display_competition,
                analysis_callback=self._on_vision_analysis,
                assistant_status_getter=self.session_state.build_assistant_status,
                external_stop_event=self.stop_event,
            )
            return {
                "status": "ok",
                "vision_result": result,
                "final_snapshot": self.session_state.snapshot(),
            }
        finally:
            self.stop_event.set()
            if self._input_thread is not None and self._input_thread.is_alive():
                self._input_thread.join(timeout=1.0)
            self.voice_service.shutdown()
            if self._owns_console_stream and self._console_stream is not None:
                try:
                    self._console_stream.close()
                except Exception:
                    pass

    def _control_loop(self) -> None:
        bind_current_thread(
            os.getenv("SPACEMIT_VOICE_CPUSET", ""),
            logger=self.logger,
            label="multimodal control loop",
        )
        while not self.stop_event.is_set():
            command = self._poll_command()
            if command is None:
                continue
            if command in {"q", "quit", "exit"}:
                self.event_bus.publish("system.mode", "Operator requested exit", {"mode": "Shutdown"})
                self.stop_event.set()
                return
            if command in {"", "a"}:
                self._run_audio_turn()
                continue
            if command == "t":
                self._run_text_turn()
                continue
            if command == "v":
                self._run_visual_summary_turn()
                continue
            if command == "s":
                snapshot_path = self.config.vision.output_dir / "competition_display_snapshot.png"
                self.event_bus.publish(
                    "system.info",
                    f"Latest competition snapshot path: {snapshot_path}",
                    {},
                )
                continue
            self.event_bus.publish("system.info", f"Unknown command: {command}", {})

    def _redraw_dashboard(self) -> None:
        if self._dashboard_enabled:
            self.dashboard.redraw(self.session_state.snapshot())

    def _poll_command(self) -> str | None:
        if self._dashboard_enabled:
            self._redraw_dashboard()
        self._write_console("multimodal> ")
        ready, _, _ = select.select([sys.stdin], [], [], 1.0)
        if not ready:
            return None
        try:
            return sys.stdin.readline().strip().lower()
        except KeyboardInterrupt:
            self.stop_event.set()
            return "q"

    def _run_audio_turn(self) -> None:
        self.event_bus.publish("system.mode", "Enter expert QA mode", {"mode": "Expert QA"})
        try:
            user_text = self.voice_service.capture_console_user_text(auto_start_recording=True)
        except Exception as exc:
            self.event_bus.publish(
                "system.info",
                f"ASR failed: {type(exc).__name__}",
                {"error": str(exc)},
            )
            self.event_bus.publish("system.mode", "Return to inspection mode", {"mode": "Inspection"})
            return
        if user_text is None:
            self.event_bus.publish("system.info", "Audio turn cancelled", {})
            self.event_bus.publish("system.mode", "Return to inspection mode", {"mode": "Inspection"})
            return
        if not user_text.strip():
            self.event_bus.publish("system.info", "ASR result is empty", {})
            self.event_bus.publish("system.mode", "Return to inspection mode", {"mode": "Inspection"})
            return
        context_hint = self._select_voice_context(user_text)
        result = self.voice_service.process_text_turn(
            user_text,
            context_hint=context_hint or None,
            console_output=False,
        )
        self._publish_voice_turn(result, context_hint)
        self.event_bus.publish("system.mode", "Return to inspection mode", {"mode": "Inspection"})

    def _run_text_turn(self) -> None:
        self.event_bus.publish("system.mode", "Enter expert QA mode", {"mode": "Expert QA"})
        self._redraw_dashboard()
        try:
            self._write_console("Question> ")
            user_text = sys.stdin.readline().strip()
        except KeyboardInterrupt:
            user_text = ""
        if not user_text:
            self.event_bus.publish("system.info", "No text question was provided", {})
            self.event_bus.publish("system.mode", "Return to inspection mode", {"mode": "Inspection"})
            return
        context_hint = self._select_voice_context(user_text)
        result = self.voice_service.process_text_turn(
            user_text,
            context_hint=context_hint or None,
            console_output=False,
        )
        self._publish_voice_turn(result, context_hint)
        self.event_bus.publish("system.mode", "Return to inspection mode", {"mode": "Inspection"})

    def _run_visual_summary_turn(self) -> None:
        context_hint = self.session_state.build_visual_context()
        if not context_hint:
            self.event_bus.publish(
                "system.info",
                "No stable visual result is available yet, so summary playback is skipped.",
                {},
            )
            return
        self.event_bus.publish("system.mode", "Enter summary mode", {"mode": "Summary"})
        summary_prompt = (
            "Please answer in Chinese. First summarize the current scene in two sentences, "
            "then give one conservative field recommendation."
        )
        result = self.voice_service.process_text_turn(
            summary_prompt,
            context_hint=context_hint,
            console_output=False,
        )
        self._publish_voice_turn(result, context_hint)
        self.event_bus.publish("system.mode", "Return to inspection mode", {"mode": "Inspection"})

    def _select_voice_context(self, user_text: str) -> str:
        context_hint = self.session_state.build_visual_context()
        if not context_hint:
            return ""

        normalized = user_text.strip().lower()
        if not normalized:
            return ""
        if any(term in normalized for term in self._VISUAL_CONTEXT_STRONG_TERMS):
            return context_hint
        if any(term in normalized for term in self._VISUAL_CONTEXT_DEICTIC_TERMS) and any(
            term in normalized for term in self._VISUAL_CONTEXT_OBJECT_TERMS
        ):
            return context_hint
        return ""

    def _publish_voice_turn(self, result: VoiceTurnResult, context_hint: str) -> None:
        self.event_bus.publish(
            "voice.turn",
            "Completed one multimodal turn",
            {
                "user_text": result.user_text,
                "reply_text": result.reply_text,
                "citations": list(result.citations),
                "metrics": {
                    "first_chunk_ms": result.first_chunk_ms,
                    "first_tts_enqueue_ms": result.first_tts_enqueue_ms,
                    "total_ms": result.total_ms,
                    "output_chars": result.output_chars,
                    "rag_used": result.rag_used,
                    "visual_context_used": bool(context_hint.strip()),
                },
            },
        )

    def _on_voice_status(self, payload: dict[str, Any]) -> None:
        headline = str(payload.get("headline", "")).strip() or "Voice status updated"
        self.event_bus.publish("voice.status", headline, payload)

    def _on_vision_analysis(self, payload: dict[str, Any]) -> None:
        summary = "Vision pipeline updated"
        result_dict = payload.get("result_dict") or {}
        candidates = list(result_dict.get("candidates", []))
        if candidates:
            top = candidates[0]
            summary = f"Top visual candidate: {top.get('label', 'unknown')} {float(top.get('score', 0.0)):.2f}"
        self.event_bus.publish("vision.analysis", summary, payload)

    def _resolve_console_stream(self) -> tuple[TextIO | None, bool]:
        if sys.stdout.isatty():
            return sys.stdout, False
        if not self._interactive_console:
            return None, False
        tty_path = os.environ.get("MULTIMODAL_TUI_TTY", "").strip() or "/dev/tty"
        try:
            stream = open(tty_path, "w", encoding="utf-8", buffering=1)
        except OSError:
            return None, False
        return stream, True

    def _write_console(self, text: str) -> None:
        if self._console_stream is None:
            return
        self._console_stream.write(text)
        self._console_stream.flush()
