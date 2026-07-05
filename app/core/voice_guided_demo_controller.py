from __future__ import annotations

import logging
import os
import re
import threading
import time
import wave
from pathlib import Path
from typing import Any

from app.core.config import AppConfig
from app.core.cpu_affinity import bind_current_thread
from app.core.event_bus import EventBus
from app.core.session_state import MultimodalSessionState
from app.ui.terminal_dashboard import TerminalDashboard
from app.vision.selector_display import SelectorDisplay
from app.vision.service import VisionPipelineService
from app.voice.passive_listener import PassiveAudioListener, PassiveUtterance
from app.voice.service import ResidentVoiceService, VoiceTurnResult


class VoiceGuidedDemoController:
    WAKE_WORD = "你好"
    WAKE_WORD_PATTERN = re.compile(r"你[\s,，。！？!?:：;；、]*好")
    EXIT_PROGRAM_TERMS = (
        "退出程序",
        "关闭程序",
        "结束程序",
        "退出系统",
        "关闭系统",
        "结束演示",
        "退出演示",
        "停止演示",
        "结束当前程序",
        "退出当前程序",
        "退出这个程序",
        "关闭这个程序",
        "停止程序",
        "停掉程序",
        "停止系统",
        "关闭应用",
        "退出应用",
    )
    CAMERA_ONE_SWITCH_TERMS = (
        "进入相机一",
        "切换相机一",
        "切换到相机一",
        "切到相机一",
        "转到相机一",
        "跳到相机一",
        "改到相机一",
        "换到相机一",
        "切换一号相机",
        "切到一号相机",
        "切换第一路相机",
        "切到第一路相机",
        "切换相机1",
        "切到相机1",
        "相机一智能巡检",
        "一号相机智能巡检",
        "第一路相机智能巡检",
    )
    CAMERA_TWO_SWITCH_TERMS = (
        "进入相机二",
        "切换相机二",
        "切换到相机二",
        "切到相机二",
        "转到相机二",
        "跳到相机二",
        "改到相机二",
        "换到相机二",
        "切换二号相机",
        "切到二号相机",
        "切换第二路相机",
        "切到第二路相机",
        "切换相机2",
        "切到相机2",
        "相机二智能巡检",
        "二号相机智能巡检",
        "第二路相机智能巡检",
    )
    VISUAL_SUMMARY_TERMS = (
        "总结一下",
        "播报总结",
        "当前画面总结",
        "汇报一下",
        "总结当前画面",
        "说一下当前画面",
        "总结当前巡检",
        "语音总结",
        "summary",
    )
    SNAPSHOT_TERMS = (
        "保存快照",
        "保存截图",
        "截个图",
        "拍个快照",
        "保存当前画面",
        "截图",
        "快照",
        "snapshot",
    )
    SELECTOR_MODE = "Camera Select"
    USB_MODE = "USB Inspection"
    MIPI_MODE = "MIPI Inspection"
    QA_MODE = "Expert QA"
    FOLLOWUP_RESUME_DELAY_SECONDS = 0.25
    BACKGROUND_RESUME_DELAY_SECONDS = 0.40
    PASSIVE_REARM_DELAY_SECONDS = 0.40
    WAKE_CAPTURE_BLOCK_DURATION_MS = 80
    WAKE_CAPTURE_END_SILENCE_BLOCKS = 3
    WAKE_CAPTURE_MAX_SECONDS = 1.80
    WAKE_CAPTURE_ENERGY_THRESHOLD = 0.027
    WAKE_CAPTURE_START_SPEECH_BLOCKS = 4
    WAKE_CAPTURE_MIN_SPEECH_BLOCKS = 5
    DEFAULT_CAPTURE_BLOCK_DURATION_MS = WAKE_CAPTURE_BLOCK_DURATION_MS
    DEFAULT_CAPTURE_END_SILENCE_BLOCKS = WAKE_CAPTURE_END_SILENCE_BLOCKS
    DEFAULT_CAPTURE_MAX_SECONDS = WAKE_CAPTURE_MAX_SECONDS
    PASSIVE_MIN_RMS = 0.025
    # Leave a little headroom for 120 ms block rounding so normal wake captures
    # do not fall into the slower long-window rescanning path.
    PASSIVE_MAX_WAKE_SECONDS = WAKE_CAPTURE_MAX_SECONDS + 0.30
    PASSIVE_LONG_CHUNK_WINDOW_SECONDS = 2.00
    PASSIVE_LONG_CHUNK_STEP_SECONDS = 1.60
    PASSIVE_LONG_CHUNK_FALLBACK_TAIL_SECONDS = 3.00
    FOLLOWUP_CAPTURE_BLOCK_DURATION_MS = 120
    FOLLOWUP_CAPTURE_END_SILENCE_BLOCKS = 4
    FOLLOWUP_CAPTURE_MAX_SECONDS = 5.50
    FOLLOWUP_CAPTURE_ENERGY_THRESHOLD = 0.013
    FOLLOWUP_CAPTURE_START_SPEECH_BLOCKS = 2
    FOLLOWUP_CAPTURE_MIN_SPEECH_BLOCKS = 3
    PASSIVE_MIN_USEFUL_SECONDS = 0.88

    def __init__(
        self,
        config: AppConfig,
        *,
        display_competition: bool = True,
        performance_mode: bool = True,
    ):
        self.config = config
        self.logger = logging.getLogger("app.core.voice_guided_demo_controller")
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

        self.voice_service = ResidentVoiceService(config, status_hook=self._on_voice_status)
        self.vision_service = VisionPipelineService(config)
        passive_device, passive_sample_rate = self._resolve_passive_input_config()
        self.listener = PassiveAudioListener(
            self._enqueue_passive_utterance,
            sample_rate=passive_sample_rate,
            channels=1,
            block_duration_ms=self.DEFAULT_CAPTURE_BLOCK_DURATION_MS,
            energy_threshold=self.WAKE_CAPTURE_ENERGY_THRESHOLD,
            start_speech_blocks=self.WAKE_CAPTURE_START_SPEECH_BLOCKS,
            end_silence_blocks=self.DEFAULT_CAPTURE_END_SILENCE_BLOCKS,
            min_speech_blocks=self.WAKE_CAPTURE_MIN_SPEECH_BLOCKS,
            max_utterance_seconds=self.DEFAULT_CAPTURE_MAX_SECONDS,
            tail_padding_blocks=2,
            queue_blocks=160,
            latency_seconds=0.25,
            never_drop_input=True,
            reset_on_overflow=True,
            device=passive_device,
        )
        self._utterance_queue: list[PassiveUtterance] = []
        self._queue_lock = threading.RLock()
        self._voice_thread: threading.Thread | None = None
        self._vision_thread: threading.Thread | None = None
        self._selector_thread: threading.Thread | None = None
        self._followup_block_duration_ms = self.FOLLOWUP_CAPTURE_BLOCK_DURATION_MS
        self._followup_end_silence_blocks = self.FOLLOWUP_CAPTURE_END_SILENCE_BLOCKS
        self._followup_max_seconds = self.FOLLOWUP_CAPTURE_MAX_SECONDS
        self._followup_energy_threshold = self.FOLLOWUP_CAPTURE_ENERGY_THRESHOLD
        self._followup_start_speech_blocks = self.FOLLOWUP_CAPTURE_START_SPEECH_BLOCKS
        self._followup_min_speech_blocks = self.FOLLOWUP_CAPTURE_MIN_SPEECH_BLOCKS
        self._console_stream, self._owns_console_stream = self._resolve_console_stream()
        self._dashboard_enabled = self._console_stream is not None
        self.dashboard = TerminalDashboard(
            title=f"{config.project_name} Voice Guided Demo",
            output_stream=self._console_stream,
        )
        self._selector_display = SelectorDisplay(
            f"{config.project_name} Voice Guided Demo",
            snapshot_path=self.config.vision.output_dir / "voice_guided_snapshot.png",
        )
        self._selector_visible = False

        self._active_mode = self.SELECTOR_MODE
        self._return_mode = self.SELECTOR_MODE
        self._vision_stop_event: threading.Event | None = None
        self._vision_running = False
        self._vision_backend = ""
        self._vision_result: dict[str, Any] | None = None
        self._vision_ready_event: threading.Event | None = None

    def run(self) -> dict[str, Any]:
        self.event_bus.publish("system.mode", "Enter camera selection mode", {"mode": self.SELECTOR_MODE})
        self.voice_service.start_workers()
        self.voice_service.warmup()
        self.event_bus.publish(
            "system.health",
            "Voice and knowledge base warmup complete",
            self.voice_service.build_health_report(),
        )
        self._voice_thread = threading.Thread(
            target=self._voice_loop,
            name="voice-guided-loop",
            daemon=True,
        )
        self._voice_thread.start()
        self.listener.start()
        self._set_selector_waiting_state()

        try:
            while not self.stop_event.is_set():
                self._service_selector_display()
                if self._dashboard_enabled:
                    self.dashboard.redraw(self.session_state.snapshot())
                time.sleep(0.08 if self._active_mode == self.SELECTOR_MODE else 0.03)
        finally:
            self.stop_event.set()
            self.listener.stop()
            self._stop_vision()
            if self._voice_thread is not None and self._voice_thread.is_alive():
                self._voice_thread.join(timeout=2.0)
            self.voice_service.shutdown()
            if self._selector_visible:
                self._selector_display.close()
                self._selector_visible = False
            if self._owns_console_stream and self._console_stream is not None:
                try:
                    self._console_stream.close()
                except Exception:
                    pass
        return {
            "status": "ok",
            "mode": self._active_mode,
            "vision_backend": self._vision_backend,
            "vision_result": self._vision_result or {},
            "final_snapshot": self.session_state.snapshot(),
        }

    def _voice_loop(self) -> None:
        bind_current_thread(
            os.getenv("SPACEMIT_VOICE_CPUSET", ""),
            logger=self.logger,
            label="voice guided loop",
        )
        while not self.stop_event.is_set():
            utterance = self._pop_passive_utterance()
            if utterance is None:
                time.sleep(0.05)
                continue
            try:
                self._handle_passive_utterance(utterance)
            finally:
                try:
                    utterance.audio_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _handle_passive_utterance(self, utterance: PassiveUtterance) -> None:
        if utterance.rms < self.PASSIVE_MIN_RMS:
            self.logger.info(
                "Skip passive utterance due to low rms duration=%.2fs rms=%.4f threshold=%.4f",
                utterance.duration_seconds,
                utterance.rms,
                self.PASSIVE_MIN_RMS,
            )
            return
        if utterance.duration_seconds < self.PASSIVE_MIN_USEFUL_SECONDS:
            self.logger.info(
                "Skip passive utterance due to short duration duration=%.2fs threshold=%.2fs rms=%.4f",
                utterance.duration_seconds,
                self.PASSIVE_MIN_USEFUL_SECONDS,
                utterance.rms,
            )
            return
        rearm_delay = self.PASSIVE_REARM_DELAY_SECONDS
        self.voice_service._update_status(
            stage="recording",
            headline="检测到语音，正在判断是否包含唤醒词",
            detail=(
                f"已捕获 {utterance.duration_seconds:.2f}s 音频片段，"
                "正在进行唤醒词级别的快速识别"
            ),
            latest_metrics={
                "audio_seconds": round(utterance.duration_seconds, 2),
                "rms": round(utterance.rms, 4),
            },
        )
        self.listener.pause()
        self._clear_pending_utterances()
        temp_paths: list[Path] = []
        try:
            user_text = self._transcribe_passive_utterance(utterance, temp_paths=temp_paths)
            if not user_text:
                self._restore_passive_waiting_state()
                return
            self.event_bus.publish(
                "system.info",
                f"Passive ASR: {user_text}",
                {
                    "user_text": user_text,
                    "duration_seconds": utterance.duration_seconds,
                    "rms": utterance.rms,
                },
            )
            command_text = self._extract_command_after_wake_word(user_text)
            if command_text is None:
                self._restore_passive_waiting_state()
                return
            rearm_delay = self.BACKGROUND_RESUME_DELAY_SECONDS
            self._clear_pending_utterances()
            self._handle_wake_word(command_text)
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            self._resume_listener(delay_seconds=rearm_delay)

    def _extract_command_after_wake_word(self, user_text: str) -> str | None:
        match = self.WAKE_WORD_PATTERN.search(user_text)
        if match is None:
            return None
        return user_text[match.end() :].strip(" \t\r\n，,。！？!?:：;；、")

    def _handle_wake_word(self, command_text: str) -> None:
        if self._looks_like_exit_command(command_text):
            self._exit_program("收到退出程序指令，正在退出。")
            return

        if self._active_mode == self.SELECTOR_MODE:
            if command_text and self._looks_like_mode_switch(command_text):
                self._handle_selector_command(command_text)
                return

            self._speak_selector_prompt()
            followup = self._capture_followup_command("请选择智能巡检相机")
            if followup:
                self._handle_selector_command(followup)
            else:
                self._set_selector_waiting_state("未收到相机选择，请再次说“你好”后选择智能巡检相机")
            return

        self._return_mode = self._active_mode
        self.event_bus.publish(
            "system.mode",
            "Enter expert QA mode",
            {"mode": self.QA_MODE, "return_mode": self._return_mode},
        )

        if command_text and self._looks_like_mode_switch(command_text):
            self._handle_selector_command(command_text)
            return

        if command_text:
            answer = self._run_expert_qa(command_text)
            self._publish_voice_turn(answer, context_hint=self._select_voice_context(command_text))
            self._restore_previous_mode()
            return

        self._speak_qa_prompt()
        followup = self._capture_followup_command("您好，请问有什么问题？")
        if self._looks_like_mode_switch(followup):
            self._handle_selector_command(followup)
            return

        answer = self._run_expert_qa(followup)
        self._publish_voice_turn(answer, context_hint=self._select_voice_context(followup))
        self._restore_previous_mode()

    def _handle_selector_command(self, command_text: str) -> bool:
        normalized = command_text.replace(" ", "")
        if self._looks_like_exit_command(normalized):
            self._exit_program("收到退出程序指令，正在退出。")
            return True
        if any(token in normalized for token in self.CAMERA_ONE_SWITCH_TERMS):
            self._start_inspection_mode("usb_v4l2", self.USB_MODE, "已进入相机一智能巡检")
            return True
        if any(token in normalized for token in self.CAMERA_TWO_SWITCH_TERMS):
            self._start_inspection_mode("mipi_official", self.MIPI_MODE, "已进入相机二智能巡检")
            return True

        self._speak_text_safely(
            "没有识别到有效的相机选择，请说进入相机一智能巡检，或进入相机二智能巡检。",
            headline="相机选择未命中",
            detail=command_text,
        )
        self.event_bus.publish(
            "system.info",
            "Camera selection command not recognized",
            {"command": command_text},
        )
        return False

    def _looks_like_visual_summary_command(self, text: str) -> bool:
        normalized = (text or "").replace(" ", "")
        if not normalized:
            return False
        return any(term in normalized for term in self.VISUAL_SUMMARY_TERMS)

    def _looks_like_snapshot_command(self, text: str) -> bool:
        normalized = (text or "").replace(" ", "")
        if not normalized:
            return False
        return any(term in normalized for term in self.SNAPSHOT_TERMS)

    def _start_inspection_mode(self, backend: str, mode: str, spoken_text: str) -> None:
        self._stop_vision()
        self._active_mode = mode
        self._vision_backend = backend
        self._vision_result = None
        self._wait_for_selector_hidden(timeout_seconds=1.2)
        self.event_bus.publish("system.mode", f"Enter {mode}", {"mode": mode, "backend": backend})
        self._vision_stop_event = threading.Event()
        self._vision_ready_event = threading.Event()
        self._vision_thread = threading.Thread(
            target=self._run_vision_mode,
            args=(backend, self._vision_stop_event, self._vision_ready_event),
            name=f"vision-{backend}",
            daemon=True,
        )
        self._vision_thread.start()
        self._set_inspection_waiting_state(mode)
        time.sleep(0.25)
        if not self._vision_running:
            self._handle_vision_start_failure(mode, backend)
            return
        self._speak_text_safely(
            spoken_text,
            headline="正在切换巡检相机",
            detail=f"{mode} / {backend}",
        )

    def _run_vision_mode(
        self,
        backend: str,
        stop_event: threading.Event,
        ready_event: threading.Event | None,
    ) -> None:
        self._vision_running = True
        try:
            result = self.vision_service.stream_camera(
                camera_backend=backend,
                recognizer_backend=None,
                interval_seconds=0.0,
                max_frames=0,
                display_status_page=False,
                performance_mode=self.performance_mode,
                display_competition=self.display_competition,
                analysis_callback=self._on_vision_analysis,
                assistant_status_getter=self.session_state.build_assistant_status,
                external_stop_event=stop_event,
            )
            self._vision_result = result
        except Exception as exc:
            self.event_bus.publish(
                "error",
                f"Vision stream failed: {type(exc).__name__}",
                {"backend": backend, "error": str(exc)},
            )
            self.logger.exception("Vision stream failed backend=%s", backend)
        finally:
            if ready_event is not None:
                ready_event.set()
            self._vision_running = False

    def _stop_vision(self) -> None:
        if self._vision_stop_event is not None:
            self._vision_stop_event.set()
        if self._vision_thread is not None and self._vision_thread.is_alive():
            self._vision_thread.join(timeout=3.0)
        time.sleep(0.2)
        self.vision_service.release_runtime_resources()
        self._vision_thread = None
        self._vision_stop_event = None
        self._vision_ready_event = None

    def _capture_followup_command(self, prompt: str) -> str:
        self.listener.configure_vad(
            block_duration_ms=self._followup_block_duration_ms,
            end_silence_blocks=self._followup_end_silence_blocks,
            max_utterance_seconds=self._followup_max_seconds,
            energy_threshold=self._followup_energy_threshold,
            start_speech_blocks=self._followup_start_speech_blocks,
            min_speech_blocks=self._followup_min_speech_blocks,
        )
        self._resume_listener(
            delay_seconds=self.FOLLOWUP_RESUME_DELAY_SECONDS,
            restore_default_profile=False,
        )
        self.voice_service._update_status(
            stage="recording",
            headline=prompt,
            detail="已唤醒，正在持续等待你的完整问题或控制指令",
        )
        while not self.stop_event.is_set():
            utterance = self._pop_passive_utterance()
            if utterance is None:
                time.sleep(0.05)
                continue
            try:
                self.voice_service._update_status(
                    stage="recording",
                    headline="正在语音输入",
                    detail=(
                        f"已捕获 {utterance.duration_seconds:.2f}s 现场语音，"
                        "接下来开始 ASR 转写"
                    ),
                    latest_metrics={
                        "audio_seconds": round(utterance.duration_seconds, 2),
                        "rms": round(utterance.rms, 4),
                    },
                )
                text = self.voice_service.transcribe_audio_file(
                    str(utterance.audio_path),
                    update_status=True,
                ).strip()
            finally:
                try:
                    utterance.audio_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if not text:
                continue
            extracted = self._extract_command_after_wake_word(text)
            if extracted is not None:
                text = extracted.strip()
            self.event_bus.publish("system.info", f"Followup ASR: {text}", {"user_text": text})
            if text:
                if self._looks_like_exit_command(text):
                    return text
                return text
        return ""

    def _run_expert_qa(self, user_text: str) -> VoiceTurnResult:
        context_hint = self._select_voice_context(user_text)
        self._pause_listener_for_speech()
        try:
            if self._looks_like_visual_summary_command(user_text):
                visual_context = self.session_state.build_visual_context()
                if not visual_context:
                    reply_text = "当前还没有稳定的视觉结果，暂时无法播报总结。"
                    self.voice_service.speak_text(
                        reply_text,
                        headline="视觉总结暂不可用",
                        detail="等待下一轮稳定视觉结果",
                    )
                    return VoiceTurnResult(
                        user_text=user_text,
                        reply_text=reply_text,
                        first_chunk_ms=0.0,
                        first_tts_enqueue_ms=0.0,
                        total_ms=0.0,
                        output_chars=len(reply_text),
                        rag_used=False,
                        citations=[],
                    )
                summary_result = self.voice_service.process_text_turn(
                    "请用中文先用两句话总结当前巡检画面，再给出一条保守的现场建议。",
                    context_hint=visual_context,
                    console_output=False,
                )
                return VoiceTurnResult(
                    user_text=user_text,
                    reply_text=summary_result.reply_text,
                    first_chunk_ms=summary_result.first_chunk_ms,
                    first_tts_enqueue_ms=summary_result.first_tts_enqueue_ms,
                    total_ms=summary_result.total_ms,
                    output_chars=summary_result.output_chars,
                    rag_used=summary_result.rag_used,
                    citations=list(summary_result.citations),
                )
            if self._looks_like_snapshot_command(user_text):
                snapshot_path = self.config.vision.output_dir / "competition_display_snapshot.png"
                reply_text = f"当前展示快照已经保存，路径是 {snapshot_path}。"
                self.event_bus.publish(
                    "system.info",
                    f"Latest competition snapshot path: {snapshot_path}",
                    {"snapshot_path": str(snapshot_path)},
                )
                self.voice_service.speak_text(
                    reply_text,
                    headline="快照已记录",
                    detail=str(snapshot_path),
                )
                return VoiceTurnResult(
                    user_text=user_text,
                    reply_text=reply_text,
                    first_chunk_ms=0.0,
                    first_tts_enqueue_ms=0.0,
                    total_ms=0.0,
                    output_chars=len(reply_text),
                    rag_used=False,
                    citations=[],
                )
            return self.voice_service.process_text_turn(
                user_text,
                context_hint=context_hint or None,
                console_output=False,
            )
        finally:
            self._resume_listener(delay_seconds=self.BACKGROUND_RESUME_DELAY_SECONDS)

    def _run_visual_summary_turn(self) -> bool:
        context_hint = self.session_state.build_visual_context()
        if not context_hint:
            self._speak_text_safely(
                "当前还没有稳定的视觉结果，暂时无法播报总结。",
                headline="视觉总结暂不可用",
                detail="等待下一轮稳定视觉结果",
            )
            return False
        self._pause_listener_for_speech()
        try:
            result = self.voice_service.process_text_turn(
                "请用中文先用两句话总结当前巡检画面，再给出一条保守的现场建议。",
                context_hint=context_hint,
                console_output=False,
            )
        finally:
            self._resume_listener(delay_seconds=self.BACKGROUND_RESUME_DELAY_SECONDS)
        self._publish_voice_turn(result, context_hint)
        return True

    def _announce_snapshot(self) -> None:
        snapshot_path = self.config.vision.output_dir / "competition_display_snapshot.png"
        self.event_bus.publish(
            "system.info",
            f"Latest competition snapshot path: {snapshot_path}",
            {"snapshot_path": str(snapshot_path)},
        )
        self._speak_text_safely(
            "当前展示快照已经保存，可用于答辩展示与复盘。",
            headline="快照已记录",
            detail=str(snapshot_path),
        )

    def _restore_previous_mode(self, *, status_headline: str = "", status_detail: str = "") -> None:
        if self._return_mode in {self.USB_MODE, self.MIPI_MODE}:
            self._active_mode = self._return_mode
            self.event_bus.publish("system.mode", "Return to inspection mode", {"mode": self._return_mode})
            if status_headline:
                self.voice_service._update_status(
                    stage="ready",
                    headline=status_headline,
                    detail=status_detail or "已返回当前巡检模式",
                )
            else:
                self._set_inspection_waiting_state(self._return_mode)
            return

        self._active_mode = self.SELECTOR_MODE
        self.event_bus.publish("system.mode", "Return to camera selection mode", {"mode": self.SELECTOR_MODE})
        if status_headline:
            self.voice_service._update_status(
                stage="ready",
                headline=status_headline,
                detail=status_detail or "已返回相机选择状态",
            )
        else:
            self._set_selector_waiting_state()

    def _pause_listener_for_speech(self) -> None:
        self.listener.pause()
        self._clear_pending_utterances()

    def _resume_listener(self, *, delay_seconds: float, restore_default_profile: bool = True) -> None:
        if restore_default_profile:
            self.listener.configure_vad(
                block_duration_ms=self.DEFAULT_CAPTURE_BLOCK_DURATION_MS,
                end_silence_blocks=self.DEFAULT_CAPTURE_END_SILENCE_BLOCKS,
                max_utterance_seconds=self.DEFAULT_CAPTURE_MAX_SECONDS,
                energy_threshold=self.WAKE_CAPTURE_ENERGY_THRESHOLD,
                start_speech_blocks=self.WAKE_CAPTURE_START_SPEECH_BLOCKS,
                min_speech_blocks=self.WAKE_CAPTURE_MIN_SPEECH_BLOCKS,
            )
        self._clear_pending_utterances()
        self.listener.resume(delay_seconds=delay_seconds)

    def _clear_pending_utterances(self) -> None:
        with self._queue_lock:
            pending = self._utterance_queue
            self._utterance_queue = []
        for utterance in pending:
            try:
                utterance.audio_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _speak_text_safely(
        self,
        text: str,
        *,
        headline: str,
        detail: str,
        resume_listener: bool = True,
    ) -> None:
        self._pause_listener_for_speech()
        try:
            self.voice_service.speak_text(text, headline=headline, detail=detail)
        finally:
            if resume_listener:
                self._resume_listener(delay_seconds=self.BACKGROUND_RESUME_DELAY_SECONDS)

    def _set_selector_waiting_state(self, detail: str = "请先说“你好”，再选择智能巡检相机") -> None:
        self.voice_service._update_status(
            stage="listening",
            headline="等待唤醒词",
            detail=detail,
            latest_reply_text="",
        )

    def _set_inspection_waiting_state(self, mode: str) -> None:
        self.voice_service._update_status(
            stage="listening",
            headline="巡检中等待唤醒词",
            detail=f"{mode} 已启动，可随时说“你好”继续提问；若已唤醒，将持续等待你的完整问题",
        )

    def _restore_passive_waiting_state(self) -> None:
        if self._active_mode == self.SELECTOR_MODE:
            self._set_selector_waiting_state()
            return
        if self._active_mode in {self.USB_MODE, self.MIPI_MODE}:
            self._set_inspection_waiting_state(self._active_mode)

    def _resolve_passive_input_config(self) -> tuple[int | str | None, int]:
        env_device = os.getenv("SPACEMIT_PASSIVE_AUDIO_DEVICE", "").strip()
        env_rate = os.getenv("SPACEMIT_PASSIVE_AUDIO_RATE", "").strip()
        preferred_rate = int(env_rate) if env_rate.isdigit() else 16000
        if env_device:
            device: int | str
            device = int(env_device) if env_device.isdigit() else env_device
            sample_rate = preferred_rate
            self.logger.info(
                "Using passive audio input from env device=%s sample_rate=%s",
                device,
                sample_rate,
            )
            return device, sample_rate

        preferred_name_fragments = (
            "UGREEN USB MIC",
            "USB MIC-CM564",
            "CM564",
        )
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for index, device_info in enumerate(devices):
                name = str(device_info.get("name", "")).strip()
                max_input_channels = int(device_info.get("max_input_channels", 0) or 0)
                if max_input_channels < 1:
                    continue
                if not any(fragment.lower() in name.lower() for fragment in preferred_name_fragments):
                    continue
                candidate_rates: list[int] = []
                for sample_rate in (preferred_rate, 16000, 32000, 44100, 48000):
                    if sample_rate not in candidate_rates:
                        candidate_rates.append(sample_rate)
                for sample_rate in candidate_rates:
                    try:
                        sd.check_input_settings(
                            device=index,
                            samplerate=sample_rate,
                            channels=1,
                            dtype="int16",
                        )
                    except Exception:
                        continue
                    self.logger.info(
                        "Using passive audio input device index=%s name=%s sample_rate=%s",
                        index,
                        name,
                        sample_rate,
                    )
                    return index, sample_rate
        except Exception as exc:
            self.logger.warning("Passive audio input auto-detect failed: %s", exc)

        self.logger.info(
            "Using default passive audio input device with sample_rate=%s",
            preferred_rate,
        )
        return None, preferred_rate

    def _transcribe_passive_utterance(
        self,
        utterance: PassiveUtterance,
        *,
        temp_paths: list[Path],
    ) -> str:
        if utterance.duration_seconds <= self.PASSIVE_MAX_WAKE_SECONDS:
            return self.voice_service.transcribe_audio_file(
                str(utterance.audio_path),
                update_status=False,
            ).strip()

        self.logger.info(
            "Passive utterance is long duration=%.2fs rms=%.4f, scanning wake-word windows window=%.2fs step=%.2fs",
            utterance.duration_seconds,
            utterance.rms,
            self.PASSIVE_LONG_CHUNK_WINDOW_SECONDS,
            self.PASSIVE_LONG_CHUNK_STEP_SECONDS,
        )
        candidate_paths = self._build_passive_long_chunk_windows(utterance.audio_path)
        temp_paths.extend(path for path, _ in candidate_paths)

        best_text = ""
        best_chars = 0
        for index, (candidate_path, label) in enumerate(candidate_paths, start=1):
            text = self.voice_service.transcribe_audio_file(
                str(candidate_path),
                update_status=False,
            ).strip()
            self.logger.info(
                "Passive wake scan window=%s index=%s/%s text_chars=%s text=%s",
                label,
                index,
                len(candidate_paths),
                len(text),
                text or "<empty>",
            )
            if len(text) > best_chars:
                best_text = text
                best_chars = len(text)
            if self._extract_command_after_wake_word(text) is not None:
                return text
        return best_text

    def _build_passive_long_chunk_windows(self, audio_path: Path) -> list[tuple[Path, str]]:
        with wave.open(str(audio_path), "rb") as rf:
            channels = rf.getnchannels()
            sample_width = rf.getsampwidth()
            frame_rate = rf.getframerate() or 1
            total_frames = rf.getnframes()
            all_frames = rf.readframes(total_frames)

        window_frames = max(1, int(frame_rate * self.PASSIVE_LONG_CHUNK_WINDOW_SECONDS))
        step_frames = max(1, int(frame_rate * self.PASSIVE_LONG_CHUNK_STEP_SECONDS))
        bytes_per_frame = max(1, channels * sample_width)

        tail_frames = min(total_frames, int(frame_rate * self.PASSIVE_LONG_CHUNK_FALLBACK_TAIL_SECONDS))
        tail_start = max(0, total_frames - tail_frames)
        tail_label = f"{tail_start / frame_rate:.2f}-{total_frames / frame_rate:.2f}s-tail"
        segments: list[tuple[int, int, str]] = [(tail_start, total_frames, tail_label)]

        max_start = max(0, total_frames - window_frames)
        start = max_start
        while start >= 0:
            end = min(total_frames, start + window_frames)
            segments.append((start, end, f"{start / frame_rate:.2f}-{end / frame_rate:.2f}s"))
            start -= step_frames
        if total_frames <= window_frames:
            segments.append((0, total_frames, f"0.00-{total_frames / frame_rate:.2f}s"))

        deduped: list[tuple[int, int, str]] = []
        seen_ranges: set[tuple[int, int]] = set()
        for start_frame, end_frame, label in segments:
            key = (start_frame, end_frame)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            deduped.append((start_frame, end_frame, label))

        paths: list[tuple[Path, str]] = []
        for start_frame, end_frame, label in deduped:
            frame_bytes = all_frames[start_frame * bytes_per_frame : end_frame * bytes_per_frame]
            paths.append(
                (
                    self._write_temp_wav(
                        frame_bytes,
                        channels=channels,
                        sample_width=sample_width,
                        frame_rate=frame_rate,
                    ),
                    label,
                )
            )
        return paths

    def _trim_wav_tail(self, audio_path: Path, *, tail_seconds: float) -> Path:
        tail_seconds = max(0.5, float(tail_seconds))
        with wave.open(str(audio_path), "rb") as rf:
            channels = rf.getnchannels()
            sample_width = rf.getsampwidth()
            frame_rate = rf.getframerate() or 1
            total_frames = rf.getnframes()
            keep_frames = min(total_frames, int(frame_rate * tail_seconds))
            start_frame = max(0, total_frames - keep_frames)
            rf.setpos(start_frame)
            frames = rf.readframes(keep_frames)

        import tempfile

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        trimmed_path = Path(temp_wav.name)
        with wave.open(str(trimmed_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(frame_rate)
            wf.writeframes(frames)
        return trimmed_path

    def _write_temp_wav(
        self,
        frames: bytes,
        *,
        channels: int,
        sample_width: int,
        frame_rate: int,
    ) -> Path:
        import tempfile

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        temp_path = Path(temp_wav.name)
        with wave.open(str(temp_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(frame_rate)
            wf.writeframes(frames)
        return temp_path

    def _speak_selector_prompt(self) -> None:
        self._speak_text_safely(
            "请选择智能巡检相机",
            headline="相机选择提示",
            detail="请说进入相机一智能巡检，或进入相机二智能巡检",
            resume_listener=False,
        )

    def _speak_qa_prompt(self) -> None:
        self._speak_text_safely(
            "您好，请问有什么问题？",
            headline="正在进入专家问答",
            detail="等待用户提出现场问题",
            resume_listener=False,
        )

    def _enqueue_passive_utterance(self, utterance: PassiveUtterance) -> None:
        with self._queue_lock:
            self._utterance_queue.append(utterance)

    def _pop_passive_utterance(self) -> PassiveUtterance | None:
        with self._queue_lock:
            if not self._utterance_queue:
                return None
            return self._utterance_queue.pop(0)

    def _service_selector_display(self) -> None:
        if self._active_mode == self.SELECTOR_MODE and not self._vision_running:
            self._show_selector_screen()
            return
        if self._selector_visible:
            self._selector_display.close()
            self._selector_visible = False

    def _show_selector_screen(self) -> None:
        if self._active_mode != self.SELECTOR_MODE or self._vision_running:
            return
        self._selector_visible = True
        key = self._selector_display.show(
            assistant_status=self.session_state.build_assistant_status(),
        )
        if key == 27:
            self.stop_event.set()

    def _wait_for_selector_hidden(self, timeout_seconds: float) -> None:
        deadline = time.perf_counter() + max(0.0, float(timeout_seconds))
        while self._selector_visible and time.perf_counter() < deadline:
            time.sleep(0.05)

    def _looks_like_mode_switch(self, text: str) -> bool:
        normalized = text.replace(" ", "")
        return any(token in normalized for token in self.CAMERA_ONE_SWITCH_TERMS + self.CAMERA_TWO_SWITCH_TERMS)

    def _looks_like_exit_command(self, text: str) -> bool:
        normalized = (text or "").replace(" ", "")
        if not normalized:
            return False
        return any(term in normalized for term in self.EXIT_PROGRAM_TERMS)

    def _exit_program(self, spoken_text: str) -> None:
        self.event_bus.publish("system.mode", "Exit voice guided demo", {"mode": self._active_mode})
        self._speak_text_safely(
            spoken_text,
            headline="正在退出程序",
            detail=spoken_text,
            resume_listener=False,
        )
        self.stop_event.set()

    def _select_voice_context(self, user_text: str) -> str:
        from app.core.multimodal_controller import MultimodalDemoController

        context_hint = self.session_state.build_visual_context()
        if not context_hint:
            return ""
        normalized = user_text.strip().lower()
        if not normalized:
            return ""
        if any(term in normalized for term in MultimodalDemoController._VISUAL_CONTEXT_STRONG_TERMS):
            return context_hint
        if any(term in normalized for term in MultimodalDemoController._VISUAL_CONTEXT_DEICTIC_TERMS) and any(
            term in normalized for term in MultimodalDemoController._VISUAL_CONTEXT_OBJECT_TERMS
        ):
            return context_hint
        return ""

    def _publish_voice_turn(self, result: VoiceTurnResult, context_hint: str) -> None:
        rag_hits = list((self.session_state.build_assistant_status() or {}).get("latest_rag_hits", []))
        self.event_bus.publish(
            "voice.turn",
            "Completed one multimodal turn",
            {
                "user_text": result.user_text,
                "reply_text": result.reply_text,
                "citations": list(result.citations),
                "rag_hits": rag_hits,
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
        if self._vision_ready_event is not None and not self._vision_ready_event.is_set():
            self._vision_ready_event.set()
        summary = "Vision pipeline updated"
        result_dict = payload.get("result_dict") or {}
        candidates = list(result_dict.get("candidates", []))
        if candidates:
            top = candidates[0]
            summary = f"Top visual candidate: {top.get('label', 'unknown')} {float(top.get('score', 0.0)):.2f}"
        self.event_bus.publish("vision.analysis", summary, payload)

    def _wait_for_vision_ready(self, timeout_seconds: float) -> None:
        ready_event = self._vision_ready_event
        if ready_event is None:
            return
        try:
            ready_event.wait(timeout=max(0.0, float(timeout_seconds)))
        except Exception:
            return

    def _handle_vision_start_failure(self, mode: str, backend: str) -> None:
        self._active_mode = self.SELECTOR_MODE
        self._vision_backend = ""
        self._vision_result = None
        self.event_bus.publish(
            "system.mode",
            "Return to camera selection mode after vision start failure",
            {"mode": self.SELECTOR_MODE, "failed_mode": mode, "backend": backend},
        )
        self._set_selector_waiting_state(
            f"{mode} 启动失败，请再次说“你好”后重新选择智能巡检相机"
        )

    def _resolve_console_stream(self):
        import sys

        if sys.stdout.isatty():
            return sys.stdout, False
        tty_path = os.environ.get("MULTIMODAL_TUI_TTY", "").strip() or "/dev/tty"
        try:
            stream = open(tty_path, "w", encoding="utf-8", buffering=1)
        except OSError:
            return None, False
        return stream, True
