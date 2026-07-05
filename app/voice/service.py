from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
import wave
from dataclasses import asdict, dataclass

from app.core.config import AppConfig
from app.core.cpu_affinity import (
    affinity_scope,
    bind_current_thread,
    bind_process,
    find_processes_by_cmdline,
    format_cpuset,
)
from app.rag import LocalKnowledgeBase, RetrievedChunk
from app.rag.query_guard import contains_domain_signal, filter_strong_hits, query_focus_terms
from app.ui import StatusPageWriter
from app.voice.audio_io import AudioPlayer, ConsoleAudioRecorder, split_tts_segments


@dataclass
class VoiceTurnResult:
    user_text: str
    reply_text: str
    first_chunk_ms: float
    first_tts_enqueue_ms: float
    total_ms: float
    output_chars: int
    rag_used: bool
    citations: list[str]


class ResidentVoiceService:
    _VISUAL_DIRECT_ACTION_TERMS = (
        "怎么办",
        "怎么处理",
        "如何处理",
        "怎么做",
        "如何做",
        "怎么修",
        "如何修",
        "先做什么",
        "下一步",
        "建议",
        "注意什么",
        "临时措施",
        "要点",
    )
    _VISUAL_DIRECT_BLOCK_TERMS = (
        "哪种",
        "什么腐蚀",
        "是什么",
        "像什么",
        "属于什么",
        "什么类型",
        "是不是",
        "怎么判断",
        "判断什么",
        "更像",
    )
    _VISUAL_IDENTIFY_TERMS = (
        "哪种",
        "什么腐蚀",
        "是什么",
        "像什么",
        "属于什么",
        "什么类型",
        "是不是",
        "更像",
        "危险吗",
        "风险高吗",
        "严不严重",
        "严重吗",
    )
    _VISUAL_RISK_TERMS = (
        "危险吗",
        "风险高吗",
        "严不严重",
        "严重吗",
    )
    _VISUAL_TOP_CANDIDATE_RE = re.compile(
        r"目标1：(?P<label>[^，。\n]+)，置信度\s*(?P<score>\d+(?:\.\d+)?)"
    )

    def __init__(self, config: AppConfig, status_hook=None):
        self.config = config
        self.logger = logging.getLogger("voice.service")
        self._status_hook = status_hook
        self.metrics_path = config.logging.turns_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.rag = LocalKnowledgeBase(config.rag)
        self.status_page = self._create_status_page()

        self._bootstrap_runtime_env()
        self.recorder = ConsoleAudioRecorder()
        self.player = AudioPlayer(
            device=config.voice.playback_device,
            volume=config.voice.playback_volume,
            sample_rate=config.voice.playback_sample_rate,
            channels=config.voice.playback_channels,
        )

        self.tts_text_queue: queue.Queue = queue.Queue()
        self.audio_queue: queue.Queue = queue.Queue()
        self._workers_started = False
        self._shutdown_requested = False
        self._voice_cpuset = os.getenv("SPACEMIT_VOICE_CPUSET", "").strip()
        self._tts_cpuset = os.getenv("SPACEMIT_TTS_CPUSET", self._voice_cpuset).strip()
        self._asr_cpuset = os.getenv(
            "SPACEMIT_ASR_CPUSET",
            self._tts_cpuset or self._voice_cpuset,
        ).strip()
        self._ollama_cpuset = os.getenv("SPACEMIT_OLLAMA_CPUSET", self._voice_cpuset).strip()
        self._ollama_affinity_thread = None
        self._pinned_ollama_pids: set[int] = set()

        self.asr_model = None
        self.llm_model = None
        self.general_llm_model = None
        self.tts_model = None
        self._tts_thread = None
        self._play_thread = None

        self._update_status(
            stage="booting",
            headline="正在加载运行时",
            detail="准备 ASR / LLM / TTS / RAG 板端服务",
            latest_metrics={"workers_started": False},
            latest_rag_hits=[],
        )
        self._load_runtimes()
        self._update_status(
            stage="idle",
            headline="运行时已就绪",
            detail="等待 warmup、doctor 或语音问答",
            latest_metrics={"workers_started": False, "rag_enabled": self.config.rag.enabled},
            latest_rag_hits=[],
        )

    def _create_status_page(self) -> StatusPageWriter | None:
        if not self.config.ui.enabled:
            return None

        try:
            return StatusPageWriter(
                config=self.config.ui,
                project_name=self.config.project_name,
                llm_model=self.config.voice.llm.model,
                rag_enabled=self.config.rag.enabled,
            )
        except Exception as exc:
            self.logger.warning("Status page disabled due to init error: %s", exc)
            return None

    def _bootstrap_runtime_env(self):
        llm = self.config.voice.llm
        tts = self.config.voice.tts

        env_values = {
            "SPACEMIT_LLM_MODEL": llm.model,
            "SPACEMIT_GENERAL_LLM_MODEL": llm.model,
            "SPACEMIT_OLLAMA_API": llm.api_url,
            "SPACEMIT_SYSTEM_PROMPT": llm.system_prompt,
            "SPACEMIT_MAX_TOKENS": str(llm.max_tokens),
            "SPACEMIT_TEMPERATURE": str(llm.temperature),
            "SPACEMIT_NUM_CTX": str(llm.num_ctx),
            "SPACEMIT_NUM_THREAD": str(llm.num_thread),
            "SPACEMIT_KEEP_ALIVE": llm.keep_alive,
            "SPACEMIT_LLM_TIMEOUT": str(llm.timeout),
            "SPACEMIT_MAX_CHARS": str(llm.max_chars),
            "SPACEMIT_MIN_CHARS": str(llm.min_chars),
            "SPACEMIT_STOP_AFTER_FIRST_SENTENCE": "1"
            if llm.stop_after_first_sentence
            else "0",
            "SPACEMIT_MATCHA_ROOT": str(tts.matcha_root),
            "SPACEMIT_MATCHA_MODEL_DIR": str(tts.model_dir),
            "SPACEMIT_MATCHA_PROVIDER": tts.provider,
            "SPACEMIT_MATCHA_DEFAULT_PRESET": tts.default_preset,
            "SPACEMIT_MATCHA_MIXED_PRESET": tts.mixed_preset,
            "SPACEMIT_MATCHA_CHINESE_MODE": tts.chinese_mode,
            "SPACEMIT_MATCHA_SPEED": str(tts.speed),
            "SPACEMIT_MATCHA_VOLUME": str(tts.volume),
            "SPACEMIT_MATCHA_THREADS": str(tts.threads),
            "SPACEMIT_MATCHA_ENABLE_WARMUP": "1" if tts.enable_warmup else "0",
            "SPACEMIT_MATCHA_ALLOW_CPU_FALLBACK": "1"
            if tts.allow_cpu_fallback
            else "0",
            "SPACEMIT_MATCHA_PRELOAD_MIXED": "1" if tts.preload_mixed_engine else "0",
            "SPACEMIT_MATCHA_WARMUP_TEXT_ZH": tts.warmup_text_zh,
            "SPACEMIT_MATCHA_WARMUP_TEXT_MIXED": tts.warmup_text_mixed,
            "SPACEMIT_TTS_TRACE": "1" if tts.trace_init else "0",
        }
        os.environ.update(env_values)

    def _load_runtimes(self):
        from app.voice.llm_runtime import LlmModel
        from app.voice.matcha_tts import TTSModel

        asr_cfg = self.config.voice.asr

        self.logger.info("Loading resident ASR/LLM/TTS runtimes")
        self.llm_model = LlmModel(self.config.voice.llm.model)
        self.tts_model = TTSModel()
        from app.voice.asr_runtime import AsrModel

        self.asr_model = AsrModel(
            model_dir=str(asr_cfg.model_dir),
            prefer_optimized_model=asr_cfg.prefer_optimized_model,
            batch_size=asr_cfg.batch_size,
            language=asr_cfg.language,
            use_itn=asr_cfg.use_itn,
            intra_op_num_threads=asr_cfg.intra_op_num_threads,
        )

    def start_workers(self):
        if self._workers_started:
            return

        self._tts_thread = threading.Thread(
            target=self._tts_worker,
            name="resident-tts-worker",
            daemon=True,
        )
        self._play_thread = threading.Thread(
            target=self._play_audio_worker,
            name="resident-play-worker",
            daemon=True,
        )
        if self._ollama_cpuset:
            self._ollama_affinity_thread = threading.Thread(
                target=self._ollama_affinity_worker,
                name="ollama-affinity-worker",
                daemon=True,
            )
        self._tts_thread.start()
        self._play_thread.start()
        if self._ollama_affinity_thread is not None:
            self._ollama_affinity_thread.start()
        self._workers_started = True
        self.logger.info("Resident audio workers started")
        self._update_status(
            stage="workers_ready",
            headline="音频队列已启动",
            detail="等待预热完成或新一轮语音输入",
            latest_metrics={"workers_started": True},
        )

    def warmup(self):
        self._update_status(
            stage="warming",
            headline="正在预热",
            detail="预热 RAG、LLM 与 TTS，降低首轮响应等待",
        )

        with affinity_scope(self._voice_cpuset, logger=self.logger, label="voice warmup"):
            if self.config.rag.enabled:
                try:
                    self.rag.search("防腐巡检")
                except Exception as exc:
                    self.logger.warning("RAG warmup skipped: %s", exc)

            if self.config.voice.prewarm_llm:
                self.logger.info("Warming up LLM")
                self._pin_ollama_processes()
                try:
                    for _ in self.llm_model.generate(self.config.voice.llm.warmup_prompt):
                        pass
                except Exception as exc:
                    self.logger.warning("LLM warmup skipped: %s", exc)

        if self.config.voice.prewarm_tts:
            self.logger.info("Warming up TTS")
            with affinity_scope(self._tts_cpuset, logger=self.logger, label="tts warmup"):
                try:
                    self.tts_model.prewarm()
                    for preset_name, meta in self.tts_model.get_engine_status().items():
                        self.logger.info(
                            "TTS warmup done preset=%s provider=%s init=%sms prewarm=%sms",
                            preset_name,
                            meta.get("provider"),
                            meta.get("init_wall_ms"),
                            meta.get("prewarm_wall_ms"),
                        )
                except Exception as exc:
                    self.logger.warning("TTS warmup skipped: %s", exc)

        self._update_status(
            stage="ready",
            headline="预热完成",
            detail="板端语音与知识库服务可直接进入问答",
            latest_metrics={"workers_started": self._workers_started, "rag_enabled": self.config.rag.enabled},
        )

    def _ollama_affinity_worker(self):
        bind_current_thread(
            self._voice_cpuset or self._ollama_cpuset,
            logger=self.logger,
            label="ollama affinity worker",
        )
        while not self._shutdown_requested:
            self._pin_ollama_processes()
            time.sleep(0.5)

    def _pin_ollama_processes(self):
        if not self._ollama_cpuset:
            return

        for fragments in (("ollama", "serve"), ("ollama", "runner")):
            for pid, cmdline in find_processes_by_cmdline(fragments):
                actual = bind_process(pid, self._ollama_cpuset)
                if actual is None or pid in self._pinned_ollama_pids:
                    continue
                self._pinned_ollama_pids.add(pid)
                self.logger.info(
                    "Pinned ollama process pid=%s cpuset=%s cmd=%s",
                    pid,
                    format_cpuset(actual),
                    cmdline,
                )

    def run_console(self):
        self.start_workers()
        self.warmup()

        print(f"Using LLM model: {self.config.voice.llm.model}")
        while not self._shutdown_requested:
            self._update_status(
                stage="listening",
                headline="等待用户输入",
                detail="控制台模式下等待音频文件或新的语音问题",
            )
            user_text = self.capture_console_user_text()
            if user_text is None:
                print("Bye")
                break

            if not user_text.strip():
                print("User: <empty>")
                self.logger.info("Skip empty ASR result")
                self._update_status(
                    stage="listening",
                    headline="收到空白输入",
                    detail="本轮 ASR 结果为空，继续等待下一轮输入",
                )
                continue

            print("User:", user_text)
            print("AI:")
            result = self.process_text_turn(user_text)
            print()
            print(
                "[METRIC] "
                f"first_chunk={result.first_chunk_ms}ms | "
                f"first_tts_enqueue={result.first_tts_enqueue_ms}ms | "
                f"total={result.total_ms}ms | output_chars={result.output_chars}"
            )

    def capture_console_user_text(self, auto_start_recording: bool = False) -> str | None:
        bind_current_thread(self._voice_cpuset)
        record_started_at = time.perf_counter()
        self._update_status(
            stage="recording",
            headline="正在录音",
            detail="按下 Enter 开始录音，再次 Enter 结束录音",
        )
        audio_file = self.recorder.record_once(auto_start=auto_start_recording)
        record_wall_ms = round((time.perf_counter() - record_started_at) * 1000, 2)
        if audio_file == "":
            self.logger.info("Voice recording cancelled after %sms", record_wall_ms)
            self._update_status(
                stage="listening",
                headline="已取消本轮语音输入",
                detail="返回巡检模式，等待下一次语音或文本指令",
                latest_metrics={"record_wall_ms": record_wall_ms},
            )
            return None

        audio_seconds = -1.0
        audio_size_kb = -1.0
        try:
            audio_size_kb = round(os.path.getsize(audio_file) / 1024.0, 2)
            with wave.open(audio_file, "rb") as wav_file:
                frame_rate = wav_file.getframerate() or 1
                frame_count = wav_file.getnframes()
                audio_seconds = round(frame_count / float(frame_rate), 3)
        except Exception as exc:
            self.logger.warning("Failed to inspect recorded audio %s: %s", audio_file, exc)

        self._update_status(
            stage="transcribing",
            headline="正在语音转写",
            detail="ASR 正在将录音音频转成文本",
        )
        self.logger.info(
            "ASR transcription start record_wall=%sms audio_seconds=%ss audio_size_kb=%s path=%s",
            record_wall_ms,
            audio_seconds,
            audio_size_kb,
            audio_file,
        )
        asr_started_at = time.perf_counter()
        try:
            with affinity_scope(
                self._asr_cpuset,
                logger=self.logger,
                label="asr transcription",
            ):
                user_text = self.asr_model(audio_file)
            asr_wall_ms = round((time.perf_counter() - asr_started_at) * 1000, 2)
            self.logger.info(
                "ASR transcription done wall=%sms audio_seconds=%ss text_chars=%s",
                asr_wall_ms,
                audio_seconds,
                len(user_text.strip()),
            )
            if not user_text.strip():
                self._update_status(
                    stage="listening",
                    headline="本轮语音未识别到有效文本",
                    detail="已返回巡检模式，请靠近麦克风后重试",
                    latest_metrics={
                        "asr_wall_ms": asr_wall_ms,
                        "audio_seconds": audio_seconds,
                        "text_chars": 0,
                    },
                )
            return user_text
        except Exception as exc:
            asr_wall_ms = round((time.perf_counter() - asr_started_at) * 1000, 2)
            self.logger.exception(
                "ASR transcription failed wall=%sms audio_seconds=%ss path=%s",
                asr_wall_ms,
                audio_seconds,
                audio_file,
            )
            self._update_status(
                stage="ready",
                headline="语音转写失败",
                detail=f"ASR 运行异常：{type(exc).__name__}",
                latest_metrics={
                    "asr_error": type(exc).__name__,
                    "asr_wall_ms": asr_wall_ms,
                    "audio_seconds": audio_seconds,
                },
            )
            raise
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)

    def transcribe_audio_file(
        self,
        audio_file: str | os.PathLike[str],
        *,
        update_status: bool = True,
    ) -> str:
        bind_current_thread(self._voice_cpuset)
        audio_path = os.fspath(audio_file)
        audio_seconds = -1.0
        audio_size_kb = -1.0
        try:
            audio_size_kb = round(os.path.getsize(audio_path) / 1024.0, 2)
            with wave.open(audio_path, "rb") as wav_file:
                frame_rate = wav_file.getframerate() or 1
                frame_count = wav_file.getnframes()
                audio_seconds = round(frame_count / float(frame_rate), 3)
        except Exception as exc:
            self.logger.warning("Failed to inspect audio %s: %s", audio_path, exc)

        if update_status:
            self._update_status(
                stage="transcribing",
                headline="正在语音转写",
                detail="ASR 正在将唤醒后的语音音频转成文本",
                latest_metrics={
                    "audio_seconds": audio_seconds,
                    "audio_size_kb": audio_size_kb,
                },
            )
        self.logger.info(
            "ASR passive transcription start audio_seconds=%ss audio_size_kb=%s path=%s",
            audio_seconds,
            audio_size_kb,
            audio_path,
        )
        asr_started_at = time.perf_counter()
        with affinity_scope(
            self._asr_cpuset,
            logger=self.logger,
            label="passive asr transcription",
        ):
            user_text = self.asr_model(audio_path)
        asr_wall_ms = round((time.perf_counter() - asr_started_at) * 1000, 2)
        self.logger.info(
            "ASR passive transcription done wall=%sms audio_seconds=%ss text_chars=%s",
            asr_wall_ms,
            audio_seconds,
            len(user_text.strip()),
        )
        if update_status and not user_text.strip():
            self._update_status(
                stage="listening",
                headline="本轮语音未识别到有效文本",
                detail="已返回待命状态，请靠近麦克风后重试",
                latest_metrics={
                    "asr_wall_ms": asr_wall_ms,
                    "audio_seconds": audio_seconds,
                    "text_chars": 0,
                },
            )
        return user_text

    def speak_text(self, text: str, *, headline: str = "正在语音播报", detail: str = "") -> None:
        text = (text or "").strip()
        if not text:
            return
        self.start_workers()
        self._update_status(
            stage="speaking",
            headline=headline,
            detail=detail or text,
            latest_reply_text=text,
        )
        remaining = text
        while remaining:
            ready_segments, remaining = split_tts_segments(
                remaining,
                max_chars=self.config.voice.segment_max_chars,
                min_chars=self.config.voice.segment_min_chars,
                flush=True,
            )
            for segment in ready_segments:
                self.tts_text_queue.put({"text": segment, "enqueued_at": time.time()})
            break
        self.tts_text_queue.join()
        self.audio_queue.join()
        self._update_status(
            stage="ready",
            headline="语音播报完成",
            detail=text,
            latest_reply_text=text,
        )

    def process_text_turn(
        self,
        user_text: str,
        context_hint: str | None = None,
        console_output: bool = True,
    ) -> VoiceTurnResult:
        bind_current_thread(self._voice_cpuset)
        self._pin_ollama_processes()
        turn_start = time.time()
        first_chunk_at = None
        first_tts_enqueue_at = None
        visible_text = ""
        tts_buffer = ""
        llm_input, rag_hits, citation_labels, fallback_notice, direct_reply = self._build_answer_context(
            user_text,
            context_hint=context_hint,
        )
        hit_summaries = self._summarize_hits(rag_hits)
        response_mode = self._decide_response_mode(user_text, rag_hits, direct_reply, context_hint)
        if direct_reply is None and response_mode == "general_llm":
            direct_reply = self._build_general_realtime_reply(user_text)

        self._update_status(
            stage="thinking",
            headline="正在生成回答",
            detail=self._build_status_detail(response_mode, rag_hits),
            latest_user_text=user_text,
            latest_reply_text="",
            latest_citations=[],
            latest_metrics={"response_mode": response_mode, "retrieved_hits": len(rag_hits)},
            latest_rag_hits=hit_summaries,
        )

        if direct_reply:
            first_chunk_at = time.time()
            if console_output:
                print(direct_reply, end="", flush=True)
            visible_text = direct_reply
            tts_buffer = direct_reply
        else:
            stream = (
                self._stream_general_reply(user_text)
                if response_mode == "general_llm"
                else self.llm_model.generate(llm_input)
            )
            for chunk in stream:
                if not chunk:
                    continue

                if first_chunk_at is None:
                    first_chunk_at = time.time()

                if console_output:
                    print(chunk, end="", flush=True)
                visible_text += chunk
                tts_buffer += chunk

                ready_segments, tts_buffer = split_tts_segments(
                    tts_buffer,
                    max_chars=self.config.voice.segment_max_chars,
                    min_chars=self.config.voice.segment_min_chars,
                )
                for segment in ready_segments:
                    if first_tts_enqueue_at is None:
                        first_tts_enqueue_at = time.time()
                    if console_output:
                        print(f"\n[TTS QUEUED] {segment}")
                    self.tts_text_queue.put({"text": segment, "enqueued_at": time.time()})

        if not visible_text.strip():
            spoken_fallback = self._build_spoken_fallback(user_text, response_mode, fallback_notice, rag_hits)
            if spoken_fallback:
                if first_chunk_at is None:
                    first_chunk_at = time.time()
                if console_output:
                    print(spoken_fallback, end="", flush=True)
                visible_text = spoken_fallback
                tts_buffer = spoken_fallback

        ready_segments, tts_buffer = split_tts_segments(
            tts_buffer,
            max_chars=self.config.voice.segment_max_chars,
            min_chars=self.config.voice.segment_min_chars,
            flush=True,
        )
        for segment in ready_segments:
            if first_tts_enqueue_at is None:
                first_tts_enqueue_at = time.time()
            if console_output:
                print(f"[TTS QUEUED] {segment}")
            self.tts_text_queue.put({"text": segment, "enqueued_at": time.time()})

        if visible_text.strip():
            self._update_status(
                stage="speaking",
                headline="正在语音播报",
                detail="TTS 合成与扬声器播放中，视觉分析将保持低频刷新",
                latest_user_text=user_text,
                latest_reply_text=visible_text.strip(),
                latest_citations=citation_labels,
                latest_metrics={
                    "response_mode": response_mode,
                    "retrieved_hits": len(rag_hits),
                },
                latest_rag_hits=hit_summaries,
            )
        self.tts_text_queue.join()
        self.audio_queue.join()

        turn_end = time.time()
        reply_body = visible_text.strip()
        reply_text = reply_body
        if citation_labels:
            citation_line = "；".join(citation_labels)
            if console_output:
                print(f"\n[参考] {citation_line}")
            reply_text = f"{reply_body}\n参考：{citation_line}" if reply_body else f"参考：{citation_line}"
        elif fallback_notice:
            if console_output:
                print(f"\n[提示] {fallback_notice}")
            reply_text = f"{reply_body}\n提示：{fallback_notice}" if reply_body else f"提示：{fallback_notice}"

        result = VoiceTurnResult(
            user_text=user_text,
            reply_text=reply_text,
            first_chunk_ms=-1
            if first_chunk_at is None
            else round((first_chunk_at - turn_start) * 1000, 2),
            first_tts_enqueue_ms=-1
            if first_tts_enqueue_at is None
            else round((first_tts_enqueue_at - turn_start) * 1000, 2),
            total_ms=round((turn_end - turn_start) * 1000, 2),
            output_chars=len(reply_body),
            rag_used=bool(rag_hits),
            citations=citation_labels,
        )
        metrics = {
            "response_mode": response_mode,
            "retrieved_hits": len(rag_hits),
            "first_chunk_ms": result.first_chunk_ms,
            "first_tts_enqueue_ms": result.first_tts_enqueue_ms,
            "total_ms": result.total_ms,
            "output_chars": result.output_chars,
            "rag_used": result.rag_used,
            "visual_context_used": bool((context_hint or "").strip()),
        }

        self._write_turn_record(result)
        if self.status_page is not None:
            self.status_page.record_turn(result)
        self._update_status(
            stage="ready",
            headline="最近一轮已完成",
            detail=self._build_completion_detail(response_mode, rag_hits, result.total_ms, fallback_notice),
            latest_user_text=user_text,
            latest_reply_text=reply_text,
            latest_citations=citation_labels,
            latest_metrics=metrics,
            latest_rag_hits=hit_summaries,
        )
        self.logger.info(
            "Voice turn done first_chunk=%sms first_tts=%sms total=%sms chars=%s rag_used=%s",
            result.first_chunk_ms,
            result.first_tts_enqueue_ms,
            result.total_ms,
            result.output_chars,
            result.rag_used,
        )
        return result

    def build_health_report(self):
        return {
            "config_path": str(self.config.config_path),
            "project_root": str(self.config.paths.root_dir),
            "models_dir": str(self.config.paths.models_dir),
            "runtime_log": str(self.config.logging.runtime_path),
            "turns_log": str(self.config.logging.turns_path),
            "asr_model_dir_exists": self.config.voice.asr.model_dir.exists(),
            "asr_status": self.asr_model.get_runtime_status() if self.asr_model else {},
            "matcha_root_exists": self.config.voice.tts.matcha_root.exists(),
            "matcha_model_dir_exists": self.config.voice.tts.model_dir.exists(),
            "workers_started": self._workers_started,
            "tts_status": self.tts_model.get_engine_status() if self.tts_model else {},
            "rag_status": self.rag.get_status(),
            "ui_status": self.status_page.get_status()
            if self.status_page is not None
            else {"enabled": False},
        }

    def shutdown(self):
        self._shutdown_requested = True
        self._update_status(
            stage="stopping",
            headline="正在停止服务",
            detail="准备关闭板端语音线程与状态页输出",
        )
        if not self._workers_started:
            self._update_status(
                stage="stopped",
                headline="服务已停止",
                detail="未启动音频队列，已完成退出",
            )
            return

        self.tts_text_queue.put(None)
        self.audio_queue.put(None)
        self.tts_text_queue.join()
        self.audio_queue.join()
        if self._ollama_affinity_thread is not None and self._ollama_affinity_thread.is_alive():
            self._ollama_affinity_thread.join(timeout=1.0)
        self._update_status(
            stage="stopped",
            headline="服务已停止",
            detail="音频队列与状态页写盘已结束",
        )
        self.logger.info("Resident voice service shutdown complete")

    def _tts_worker(self):
        bind_current_thread(
            self._tts_cpuset,
            logger=self.logger,
            label="tts synthesis worker",
        )
        while True:
            item = self.tts_text_queue.get()
            if item is None:
                self.tts_text_queue.task_done()
                break

            try:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    enqueued_at = item.get("enqueued_at")
                else:
                    text = item
                    enqueued_at = None

                queue_wait_ms = None
                if enqueued_at is not None:
                    queue_wait_ms = round((time.time() - enqueued_at) * 1000, 2)

                output_audio = self.tts_model.ort_predict(text, queue_wait_ms=queue_wait_ms)
                metrics = self.tts_model.get_last_metrics()
                self.audio_queue.put((text, output_audio, metrics))
            finally:
                self.tts_text_queue.task_done()

    def _play_audio_worker(self):
        bind_current_thread(
            self._tts_cpuset,
            logger=self.logger,
            label="tts playback worker",
        )
        while True:
            item = self.audio_queue.get()
            if item is None:
                self.audio_queue.task_done()
                break

            sentence, output_audio, metrics = item
            try:
                self.logger.info("Play queued text=%s", sentence)
                if metrics:
                    self.logger.info(
                        "TTS metric preset=%s provider=%s queue_wait=%sms wall=%sms proc=%sms rtf=%s",
                        metrics.get("preset"),
                        metrics.get("provider"),
                        metrics.get("queue_wait_ms"),
                        metrics.get("synth_wall_ms"),
                        metrics.get("processing_time_ms"),
                        metrics.get("rtf"),
                    )
                self.player.play_audio(output_audio)
            except Exception as exc:
                self.logger.warning("Audio playback failed text=%s err=%s", sentence, exc)
            finally:
                if (
                    not self.config.voice.tts.keep_tts_wav
                    and output_audio
                    and os.path.exists(output_audio)
                ):
                    try:
                        os.remove(output_audio)
                    except OSError:
                        pass
                self.audio_queue.task_done()

    def _write_turn_record(self, result: VoiceTurnResult):
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            **asdict(result),
        }
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _get_general_llm_model(self):
        if self.general_llm_model is not None:
            return self.general_llm_model

        from app.voice.llm_runtime import LlmModel

        model_name = os.getenv("SPACEMIT_GENERAL_LLM_MODEL", "qwen2.5:0.5b").strip() or "qwen2.5:0.5b"
        system_prompt = (
            "你是板端通用中文助手。"
            "用户问日常通用问题时，直接给自然、简短、实用的中文回答。"
            "不要复述用户问题，不要展示思考过程。"
            "如果问题需要实时信息或你无法确认，就直接说明你现在不能核实，"
            "再给一个有帮助的下一步建议。"
        )
        self.general_llm_model = LlmModel(
            model_name,
            system_prompt=system_prompt,
            max_chars=64,
            min_chars=8,
            stop_after_first_sentence=False,
        )
        return self.general_llm_model

    def _stream_general_reply(self, user_text: str):
        try:
            model = self._get_general_llm_model()
            yield from model.generate(user_text)
        except Exception as exc:
            self.logger.warning("General LLM path failed: %s", exc)
            return

    def _build_answer_context(
        self,
        user_text: str,
        context_hint: str | None = None,
    ) -> tuple[str, list[RetrievedChunk], list[str], str | None, str | None]:
        direct_visual_reply = self._build_direct_visual_reply(user_text, context_hint or "")
        wants_visual_action = self._wants_visual_action_guidance(user_text, context_hint or "")
        if direct_visual_reply and not wants_visual_action:
            return user_text, [], [], None, direct_visual_reply

        if not self.config.rag.enabled:
            if context_hint:
                prompt = (
                    "你正在回答离线工业巡检问题。\n"
                    f"用户问题：{user_text}\n\n"
                    f"当前画面摘要：\n{context_hint}\n\n"
                    "请结合当前画面，用自然中文给出保守、简短、可执行的初步判断。"
                )
                return prompt, [], [], None, None
            return user_text, [], [], None, None

        if not context_hint and not contains_domain_signal(user_text):
            return user_text, [], [], None, None

        candidate_top_k = max(self.config.rag.top_k * 3, 8)
        hits = self.rag.search(user_text, top_k=candidate_top_k)
        strong_hits = self._rank_strong_hits(filter_strong_hits(user_text, hits), user_text)[: self.config.rag.top_k]
        visual_block = f"当前画面摘要：\n{context_hint}\n\n" if context_hint else ""
        if not strong_hits:
            if direct_visual_reply:
                return user_text, [], [], None, direct_visual_reply
            prompt = (
                "你正在回答工业防腐巡检问题。\n"
                f"用户问题：{user_text}\n\n"
                f"{visual_block}"
                "当前本地知识库没有检索到直接命中的资料。"
                "请像现场同事口头交流一样，用自然中文先直接回答最可能的判断，"
                "再补一句下一步建议，整体保持保守、简短、可执行。"
                "明确提醒需要结合现场规程和专业人员复核。"
                "不要编造标准号、检测数值和未看到的现场事实。"
            )
            fallback_notice = "当前未检索到直接命中的本地知识条目，回答偏保守，请结合现场规程和专业人员复核。"
            return prompt, [], [], fallback_notice, None

        citation_labels = [
            f"[{index}] {hit.source_label}"
            for index, hit in enumerate(strong_hits[: self.config.rag.citation_limit], start=1)
        ]

        if direct_visual_reply and wants_visual_action:
            direct_rag_reply = self._build_direct_rag_reply(strong_hits)
            top_candidate = self._parse_visual_top_candidate(context_hint or "")
            if direct_rag_reply and not self._reply_conflicts_with_visual_label(
                top_candidate[0] if top_candidate else "",
                direct_rag_reply,
            ):
                merged_reply = self._merge_visual_and_action_reply(direct_visual_reply, direct_rag_reply)
                if merged_reply:
                    return user_text, strong_hits, citation_labels, None, merged_reply
            return user_text, [], [], None, direct_visual_reply

        if strong_hits[0].score >= self.config.rag.direct_answer_score:
            direct_reply = self._build_direct_rag_reply(strong_hits)
            if direct_reply and self._reply_matches_query(user_text, direct_reply):
                if not context_hint or self._allow_direct_rag_with_visual_context(user_text, context_hint):
                    return user_text, strong_hits, citation_labels, None, direct_reply

        evidence_lines = []
        used_chars = 0
        for index, hit in enumerate(strong_hits, start=1):
            compact = " ".join(line.strip() for line in hit.text.splitlines() if line.strip())
            budget = self.config.rag.max_context_chars - used_chars
            if budget <= 0:
                break
            compact = compact[:budget].strip()
            if not compact:
                continue
            evidence_lines.append(
                f"[{index}] 标题：{hit.title}\n来源：{hit.source_label}\n内容：{compact}"
            )
            used_chars += len(compact)

        prompt = (
            f"问题：{user_text}\n\n"
            f"{visual_block}"
            "本地资料：\n"
            f"{chr(10).join(evidence_lines)}\n\n"
            "请像现场同事口头交流一样，只用一到两句自然中文，"
            "先直接回答用户最关心的点，再补一句下一步建议。"
            "不要编号，不要项目符号，不要生硬复述资料原句，"
            "不要编造标准号和未出现的现场事实。"
        )
        return prompt, strong_hits, citation_labels, None, None

    def _build_general_realtime_reply(self, user_text: str) -> str | None:
        weather_terms = ("天气", "气温", "下雨", "降温", "台风", "穿什么")
        news_terms = ("新闻", "热搜", "头条", "最新消息")
        market_terms = ("股价", "汇率", "币价", "金价", "油价")
        sports_terms = ("比分", "战绩", "赛程", "谁赢了")

        if any(term in user_text for term in weather_terms):
            return (
                "我现在没法直接查到实时天气。"
                "你告诉我城市，或者先看一下手机天气，我可以再帮你分析穿衣和出行建议。"
            )
        if any(term in user_text for term in news_terms):
            return (
                "我现在不能直接核实最新新闻。"
                "你可以给我一条你看到的消息，我帮你一起判断重点和真假。"
            )
        if any(term in user_text for term in market_terms):
            return (
                "我现在不能直接核实实时价格。"
                "你可以把你看到的价格发给我，我帮你一起分析。"
            )
        if any(term in user_text for term in sports_terms):
            return (
                "我现在不能直接核实实时比分或赛程。"
                "你把比赛双方或者你看到的结果发给我，我帮你整理。"
            )
        return None

    def _allow_direct_rag_with_visual_context(self, user_text: str, context_hint: str) -> bool:
        normalized = user_text.strip().lower()
        if not normalized or not context_hint.strip():
            return False
        if any(term in normalized for term in self._VISUAL_DIRECT_BLOCK_TERMS):
            return False
        return any(term in normalized for term in self._VISUAL_DIRECT_ACTION_TERMS)

    def _wants_visual_action_guidance(self, user_text: str, context_hint: str) -> bool:
        normalized = user_text.strip().lower()
        if not normalized or not context_hint.strip():
            return False
        return any(term in normalized for term in self._VISUAL_DIRECT_ACTION_TERMS)

    def _build_direct_visual_reply(self, user_text: str, context_hint: str) -> str | None:
        normalized = user_text.strip().lower()
        if not normalized or not context_hint.strip():
            return None
        if not any(term in normalized for term in self._VISUAL_IDENTIFY_TERMS):
            return None

        top_candidate = self._parse_visual_top_candidate(context_hint)
        if top_candidate is None:
            return None

        label, score = top_candidate
        is_risk_question = any(term in normalized for term in self._VISUAL_RISK_TERMS)

        if score >= 0.85:
            confidence_phrase = "更像"
            caution_phrase = "但还不能只凭单帧直接定型。"
        elif score >= 0.65:
            confidence_phrase = "偏向"
            caution_phrase = "不过还需要结合近距离复核再确认。"
        else:
            confidence_phrase = "目前只看到"
            caution_phrase = "现阶段更适合把它当作待复核线索。"

        if "点蚀" in label:
            if is_risk_question:
                return (
                    "结合当前画面，这里已经可以当作需要尽快复核的点蚀线索，"
                    f"{caution_phrase}建议先记录位置和范围，再检查是否存在针孔状深坑、局部积液或周边涂层破口。"
                )
            return (
                f"结合当前画面，这里{confidence_phrase}点蚀，{caution_phrase}"
                "建议先复核是否存在针孔状局部深坑，并记录位置和范围。"
            )
        if "缝隙腐蚀" in label:
            if is_risk_question:
                return (
                    "结合当前画面，这里更像需要尽快复核的缝隙腐蚀线索，"
                    f"{caution_phrase}建议优先检查搭接边、缝边、垫片附近和容易积液的夹缝位置。"
                )
            return (
                f"结合当前画面，这里{confidence_phrase}缝隙腐蚀，{caution_phrase}"
                "建议优先复核搭接边、缝边或垫片附近是否持续返锈。"
            )
        if "均匀腐蚀" in label:
            if is_risk_question:
                return (
                    "结合当前画面，这里更像需要持续复核的均匀腐蚀线索，"
                    f"{caution_phrase}建议先看减薄是否连续，再安排厚度或壁厚复核。"
                )
            return (
                f"结合当前画面，这里{confidence_phrase}均匀腐蚀，{caution_phrase}"
                "建议先复核减薄范围是否连续，再判断是否需要厚度检测。"
            )
        if "保温" in label or "cui" in label.lower():
            if is_risk_question:
                return (
                    "结合当前画面，这里已经可以当作保温层下腐蚀风险线索尽快复核，"
                    f"{caution_phrase}建议先检查保温层破损、渗水和周边锈迹，再决定是否开保温复查。"
                )
            return (
                f"结合当前画面，这里{confidence_phrase}保温层下腐蚀风险提示，{caution_phrase}"
                "建议先记录位置和范围，再结合保温层破损、渗水和周边锈迹做现场复核。"
            )
        if "剥落" in label or "分层" in label:
            if is_risk_question:
                return (
                    "结合当前画面，如果这里已经露底或边界继续扩大，就应当尽快处置，"
                    f"{caution_phrase}建议先复核附着力失效范围和基材暴露情况。"
                )
            return (
                f"结合当前画面，这里{confidence_phrase}涂层剥落疑似，{caution_phrase}"
                "建议先看是否已经露底，再复核附着力失效范围和边界。"
            )
        if "粉化" in label:
            if is_risk_question:
                return (
                    "结合当前画面，这里更像表层老化粉化线索，短期风险通常低于露底返锈，"
                    f"{caution_phrase}但仍建议先擦拭复核是否已经发展到附着力失效。"
                )
            return (
                f"结合当前画面，这里{confidence_phrase}粉化疑似，{caution_phrase}"
                "建议先擦拭复核是否有明显掉粉，再判断是否需要表面处理和重涂。"
            )
        if is_risk_question:
            return (
                "结合当前画面，这里已经可以当作需要继续复核的锈蚀线索，"
                f"{caution_phrase}建议先记录位置和范围，再检查是否伴随起泡、剥落或基材暴露。"
            )
        return (
            f"结合当前画面，这里{confidence_phrase}锈蚀疑似，{caution_phrase}"
            "建议先记录位置和范围，再复核是否伴随起泡、剥落或基材暴露。"
        )

    def _parse_visual_top_candidate(self, context_hint: str) -> tuple[str, float] | None:
        match = self._VISUAL_TOP_CANDIDATE_RE.search(context_hint)
        if match is None:
            return None

        label = match.group("label").strip()
        try:
            score = float(match.group("score"))
        except ValueError:
            score = 0.0
        return label, score

    def _merge_visual_and_action_reply(self, visual_reply: str, action_reply: str) -> str:
        merged: list[str] = []
        for sentence in self._split_sentences(visual_reply):
            cleaned = self._clean_direct_sentence(sentence)
            if not cleaned:
                continue
            merged.append(self._ensure_sentence_end(cleaned))
            break

        total_length = sum(len(part) for part in merged)
        for sentence in self._split_sentences(action_reply):
            cleaned = self._clean_direct_sentence(sentence)
            if not cleaned:
                continue
            normalized = self._ensure_sentence_end(cleaned)
            if normalized in merged:
                continue
            if merged and total_length + len(normalized) > 120:
                continue
            merged.append(normalized)
            total_length += len(normalized)
            break

        if len(merged) == 1:
            for sentence in self._split_sentences(visual_reply)[1:]:
                cleaned = self._clean_direct_sentence(sentence)
                if not cleaned:
                    continue
                normalized = self._ensure_sentence_end(cleaned)
                if normalized in merged:
                    continue
                merged.append(normalized)
                break

        return "".join(merged)

    def _reply_conflicts_with_visual_label(self, label: str, reply_text: str) -> bool:
        normalized_label = label.strip().lower()
        normalized_reply = reply_text.strip().lower()
        if not normalized_label or not normalized_reply:
            return False

        label_groups = {
            "point": ("点蚀", "pitting"),
            "crevice": ("缝隙", "crevice", "夹缝", "垫片"),
            "uniform": ("均匀腐蚀", "uniform", "减薄"),
            "cui": ("保温", "cui"),
            "flaking": ("剥落", "分层", "露底", "附着力"),
            "chalking": ("粉化", "掉粉", "失光"),
        }
        active_group = None
        for group, keywords in label_groups.items():
            if any(keyword in normalized_label for keyword in keywords):
                active_group = group
                break

        if active_group is None:
            return False

        for group, keywords in label_groups.items():
            if group == active_group:
                continue
            if any(keyword in normalized_reply for keyword in keywords):
                return True
        return False

    def _build_spoken_fallback(
        self,
        user_text: str,
        response_mode: str,
        fallback_notice: str | None,
        rag_hits: list[RetrievedChunk],
    ) -> str:
        if response_mode == "general_llm":
            return (
                "我这边刚才没顺利答出来。"
                "你可以换个更具体的说法，或者告诉我场景，我按普通常识继续回答你。"
            )

        if response_mode == "fallback_no_hit":
            return (
                "这个问题我这边暂时没有直接命中的资料，我先不给你下太死的结论。"
                "你可以补充一下部位、材质、环境和现象，我继续帮你判断。"
            )

        if response_mode == "llm_with_rag" and rag_hits:
            return (
                "我查到了一些相关资料，但这轮没有顺利组织出回答。"
                "你可以再问一遍，或者把现场情况说得更具体一点，我继续帮你看。"
            )

        if fallback_notice:
            return "我刚才没有顺利答出来，你可以换个说法再问我一次，我继续帮你看。"

        return "我刚才这轮没有顺利说出来，你再说具体一点，我继续帮你看。"

    def _rank_strong_hits(self, hits: list[RetrievedChunk], user_text: str) -> list[RetrievedChunk]:
        def penalty(hit: RetrievedChunk) -> tuple[int, float]:
            title = hit.title.strip()
            source = hit.source_label
            score_penalty = 0
            if title == "正文":
                score_penalty += 4
            if title == "检索别名":
                score_penalty += 3
            if title in {"使用边界", "问答口径建议", "现场回答建议"}:
                score_penalty += 2
            if "记录" in user_text and title == "问答口径建议":
                score_penalty -= 1
            if "问：" in title:
                score_penalty -= 2
            if source.endswith('.pdf.md）') and title == "正文":
                score_penalty += 1
            return (score_penalty, -hit.score)

        return sorted(hits, key=penalty)

    def _reply_matches_query(self, user_text: str, reply_text: str) -> bool:
        reply_lower = reply_text.lower()
        focus_terms = query_focus_terms(user_text)
        if not focus_terms:
            return False
        matched = [term for term in focus_terms if term in reply_lower]
        return len(matched) >= 1

    def _build_direct_rag_reply(self, hits: list[RetrievedChunk]) -> str | None:
        candidates: list[tuple[int, int, str]] = []
        seen = set()
        strong_keywords = ("不建议", "应先", "先看", "先查", "先确认", "建议", "优先", "可先", "需要")
        weak_keywords = ("如果", "应", "先")
        meta_phrases = ("适合回答", "不替代", "使用边界", "规则卡", "完整性评估", "当用户问", "优先从", "优先按", "不是只说", "核查")

        order = 0
        for hit in hits[:3]:
            if hit.title in {"正文", "检索别名", "使用边界", "问答口径建议", "现场回答建议"}:
                continue
            for part in self._split_sentences(hit.text):
                cleaned = self._clean_direct_sentence(part)
                if not cleaned or cleaned in seen:
                    continue
                if any(phrase in cleaned for phrase in meta_phrases):
                    continue
                seen.add(cleaned)
                priority = 0
                if any(keyword in cleaned for keyword in strong_keywords):
                    priority += 3
                elif any(keyword in cleaned for keyword in weak_keywords):
                    priority += 1
                if "回答“" in part or "回答\"" in part:
                    priority -= 2
                if len(cleaned) > 80:
                    priority -= 1
                candidates.append((priority, order, cleaned))
                order += 1

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1]))
        sentences: list[str] = []
        total_length = 0
        for _, _, cleaned in candidates:
            normalized = self._ensure_sentence_end(cleaned)
            if normalized in sentences:
                continue
            if sentences and total_length + len(normalized) > 96:
                continue
            sentences.append(normalized)
            total_length += len(normalized)
            if len(sentences) >= 2:
                break

        reply = "".join(sentences) if sentences else self._ensure_sentence_end(candidates[0][2])
        return self._clean_direct_sentence(reply)

    def _split_sentences(self, text: str) -> list[str]:
        raw_parts = []
        current = []
        for char in text.strip():
            current.append(char)
            if char in "。！？?!；":
                raw_parts.append("".join(current))
                current = []
        if current:
            raw_parts.append("".join(current))
        return [part.strip() for part in raw_parts if part.strip()]

    def _clean_direct_sentence(self, text: str) -> str:
        text = " ".join(text.split())
        text = text.replace("具体来说：", "")
        text = text.replace("具体而言：", "")
        text = text.replace("回答“", "")
        text = text.replace("”时：", "")
        text = text.replace("1.", "")
        text = text.replace("2.", "")
        text = text.replace("3.", "")
        text = text.replace("管道外壁应重点检查", "先看")
        text = text.replace("应重点检查", "先看")
        text = text.replace("重点检查", "先看")
        text = text.replace("如果腐蚀集中在", "如果返锈主要集中在")
        text = text.replace("往往与", "多半和")
        text = text.replace("保温层破损后渗水", "保温层破损渗水")
        return text.strip(" ，；")

    def _ensure_sentence_end(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        if text[-1] not in "。！？?!；":
            return text + "。"
        return text

    def _summarize_hits(self, hits: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "title": hit.title,
                "source_label": hit.source_label,
                "score": hit.score,
                "overlap_terms": hit.overlap_terms,
            }
            for hit in hits[: self.config.rag.citation_limit]
        ]

    def _decide_response_mode(
        self,
        user_text: str,
        hits: list[RetrievedChunk],
        direct_reply: str | None,
        context_hint: str | None = None,
    ) -> str:
        if direct_reply:
            if context_hint and hits:
                return "direct_visual_rag"
            if context_hint and not hits:
                return "direct_visual"
            return "direct_rag"
        if hits:
            return "llm_with_rag"
        if context_hint:
            return "fallback_no_hit"
        if self.config.rag.enabled and not contains_domain_signal(user_text):
            return "general_llm"
        if self.config.rag.enabled:
            return "fallback_no_hit"
        return "plain_llm"

    def _build_status_detail(self, response_mode: str, rag_hits: list[RetrievedChunk]) -> str:
        if response_mode == "direct_visual_rag":
            return f"当前问题依赖视觉结果，已走视觉识别 + RAG 处置双直答路径，命中 {len(rag_hits)} 条资料"
        if response_mode == "direct_visual":
            return "当前问题直接依赖稳定视觉候选，已走视觉快答路径"
        if response_mode == "direct_rag":
            return f"命中 {len(rag_hits)} 条本地知识，直接输出高置信结论并附引用"
        if response_mode == "llm_with_rag":
            return f"命中 {len(rag_hits)} 条本地知识，正在结合证据生成回答"
        if response_mode == "general_llm":
            return "检测到非防腐问题，正在按普通助手路径生成回答"
        if response_mode == "fallback_no_hit":
            return "本地知识未命中，正在走保守 fallback 回答路径"
        return "RAG 未启用，正在按普通问答路径生成回答"

    def _build_completion_detail(
        self,
        response_mode: str,
        rag_hits: list[RetrievedChunk],
        total_ms: float,
        fallback_notice: str | None,
    ) -> str:
        if response_mode == "direct_visual_rag":
            return f"视觉识别 + 知识库直答完成，命中 {len(rag_hits)} 条，本轮耗时 {total_ms} ms"
        if response_mode == "direct_visual":
            return f"视觉快答完成，本轮耗时 {total_ms} ms"
        if response_mode == "direct_rag":
            return f"知识库直答完成，命中 {len(rag_hits)} 条，本轮耗时 {total_ms} ms"
        if response_mode == "llm_with_rag":
            return f"RAG+LLM 回答完成，命中 {len(rag_hits)} 条，本轮耗时 {total_ms} ms"
        if response_mode == "general_llm":
            return f"通用问答完成，本轮耗时 {total_ms} ms"
        if response_mode == "fallback_no_hit":
            suffix = "，已附保守提示" if fallback_notice else ""
            return f"fallback 回答完成，本轮耗时 {total_ms} ms{suffix}"
        return f"普通问答完成，本轮耗时 {total_ms} ms"

    def _update_status(self, stage: str, headline: str, detail: str = "", **kwargs):
        if self.status_page is None:
            if self._status_hook is not None:
                try:
                    self._status_hook({"stage": stage, "headline": headline, "detail": detail, **kwargs})
                except Exception:
                    pass
            return
        try:
            self.status_page.update(stage=stage, headline=headline, detail=detail, **kwargs)
        except Exception as exc:
            self.logger.warning("Status page update failed: %s", exc)
            self.status_page = None
        if self._status_hook is not None:
            try:
                self._status_hook({"stage": stage, "headline": headline, "detail": detail, **kwargs})
            except Exception:
                pass
