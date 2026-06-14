from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass

from app.core.config import AppConfig
from app.rag import LocalKnowledgeBase, RetrievedChunk
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
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger("voice.service")
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

        self.asr_model = None
        self.llm_model = None
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
            "SPACEMIT_OLLAMA_API": llm.api_url,
            "SPACEMIT_SYSTEM_PROMPT": llm.system_prompt,
            "SPACEMIT_MAX_TOKENS": str(llm.max_tokens),
            "SPACEMIT_TEMPERATURE": str(llm.temperature),
            "SPACEMIT_NUM_CTX": str(llm.num_ctx),
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
        self._tts_thread.start()
        self._play_thread.start()
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

        if self.config.rag.enabled:
            try:
                self.rag.search("防腐巡检")
            except Exception as exc:
                self.logger.warning("RAG warmup skipped: %s", exc)

        if self.config.voice.prewarm_llm:
            self.logger.info("Warming up LLM")
            try:
                for _ in self.llm_model.generate(self.config.voice.llm.warmup_prompt):
                    pass
            except Exception as exc:
                self.logger.warning("LLM warmup skipped: %s", exc)

        if self.config.voice.prewarm_tts:
            self.logger.info("Warming up TTS")
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
            audio_file = self.recorder.record_once()
            if audio_file == "":
                print("Bye")
                break

            try:
                user_text = self.asr_model(audio_file)
            finally:
                if audio_file and os.path.exists(audio_file):
                    os.remove(audio_file)

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

    def process_text_turn(self, user_text: str) -> VoiceTurnResult:
        turn_start = time.time()
        first_chunk_at = None
        first_tts_enqueue_at = None
        visible_text = ""
        tts_buffer = ""
        llm_input, rag_hits, citation_labels, fallback_notice, direct_reply = self._build_answer_context(user_text)
        hit_summaries = self._summarize_hits(rag_hits)
        response_mode = self._decide_response_mode(rag_hits, direct_reply)

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
            print(direct_reply, end="", flush=True)
            visible_text = direct_reply
            tts_buffer = direct_reply
        else:
            for chunk in self.llm_model.generate(llm_input):
                if not chunk:
                    continue

                if first_chunk_at is None:
                    first_chunk_at = time.time()

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
                    print(f"\n[TTS QUEUED] {segment}")
                    self.tts_text_queue.put({"text": segment, "enqueued_at": time.time()})

        ready_segments, tts_buffer = split_tts_segments(
            tts_buffer,
            max_chars=self.config.voice.segment_max_chars,
            min_chars=self.config.voice.segment_min_chars,
            flush=True,
        )
        for segment in ready_segments:
            if first_tts_enqueue_at is None:
                first_tts_enqueue_at = time.time()
            print(f"[TTS QUEUED] {segment}")
            self.tts_text_queue.put({"text": segment, "enqueued_at": time.time()})

        self.tts_text_queue.join()
        self.audio_queue.join()

        turn_end = time.time()
        reply_body = visible_text.strip()
        reply_text = reply_body
        if citation_labels:
            citation_line = "；".join(citation_labels)
            print(f"\n[参考] {citation_line}")
            reply_text = f"{reply_body}\n参考：{citation_line}" if reply_body else f"参考：{citation_line}"
        elif fallback_notice:
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
        self._update_status(
            stage="stopped",
            headline="服务已停止",
            detail="音频队列与状态页写盘已结束",
        )
        self.logger.info("Resident voice service shutdown complete")

    def _tts_worker(self):
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

    def _build_answer_context(
        self, user_text: str
    ) -> tuple[str, list[RetrievedChunk], list[str], str | None, str | None]:
        if not self.config.rag.enabled:
            return user_text, [], [], None, None

        hits = self.rag.search(user_text, top_k=self.config.rag.top_k)
        if not hits:
            prompt = (
                "你正在回答工业防腐巡检问题。\n"
                f"用户问题：{user_text}\n\n"
                "当前本地知识库没有检索到直接命中的资料。"
                "请基于通用防腐巡检常识给出保守、简短、可执行的初判建议，"
                "明确提醒需要结合现场规程和专业人员复核。"
                "不要编造标准号、检测数值和未看到的现场事实。"
            )
            fallback_notice = "当前未检索到直接命中的本地知识条目，回答偏保守，请结合现场规程和专业人员复核。"
            return prompt, [], [], fallback_notice, None

        citation_labels = [
            f"[{index}] {hit.source_label}"
            for index, hit in enumerate(hits[: self.config.rag.citation_limit], start=1)
        ]

        if hits[0].score >= self.config.rag.direct_answer_score:
            direct_reply = self._build_direct_rag_reply(hits)
            if direct_reply:
                return user_text, hits, citation_labels, None, direct_reply

        evidence_lines = []
        used_chars = 0
        for index, hit in enumerate(hits, start=1):
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
            "本地资料：\n"
            f"{chr(10).join(evidence_lines)}\n\n"
            "请只用一到两句中文给出保守、直接、可执行的初判建议，"
            "不要编号，不要项目符号，不要编造标准号和未出现的现场事实。"
        )
        return prompt, hits, citation_labels, None, None

    def _build_direct_rag_reply(self, hits: list[RetrievedChunk]) -> str | None:
        candidates: list[tuple[int, int, str]] = []
        seen = set()
        strong_keywords = ("不建议", "应先", "先看", "先确认", "建议", "优先", "可先", "需要")
        weak_keywords = ("如果", "应", "先")

        order = 0
        for hit in hits[:3]:
            for part in self._split_sentences(hit.text):
                cleaned = self._clean_direct_sentence(part)
                if not cleaned or cleaned in seen:
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
            if cleaned in sentences:
                continue
            if sentences and total_length + len(cleaned) > 96:
                continue
            sentences.append(cleaned)
            total_length += len(cleaned)
            if len(sentences) >= 2:
                break

        reply = "".join(sentences) if sentences else candidates[0][2]
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
        return text.strip()

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

    def _decide_response_mode(self, hits: list[RetrievedChunk], direct_reply: str | None) -> str:
        if direct_reply:
            return "direct_rag"
        if hits:
            return "llm_with_rag"
        if self.config.rag.enabled:
            return "fallback_no_hit"
        return "plain_llm"

    def _build_status_detail(self, response_mode: str, rag_hits: list[RetrievedChunk]) -> str:
        if response_mode == "direct_rag":
            return f"命中 {len(rag_hits)} 条本地知识，直接输出高置信结论并附引用"
        if response_mode == "llm_with_rag":
            return f"命中 {len(rag_hits)} 条本地知识，正在结合证据生成回答"
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
        if response_mode == "direct_rag":
            return f"知识库直答完成，命中 {len(rag_hits)} 条，本轮耗时 {total_ms} ms"
        if response_mode == "llm_with_rag":
            return f"RAG+LLM 回答完成，命中 {len(rag_hits)} 条，本轮耗时 {total_ms} ms"
        if response_mode == "fallback_no_hit":
            suffix = "，已附保守提示" if fallback_notice else ""
            return f"fallback 回答完成，本轮耗时 {total_ms} ms{suffix}"
        return f"普通问答完成，本轮耗时 {total_ms} ms"

    def _update_status(self, stage: str, headline: str, detail: str = "", **kwargs):
        if self.status_page is None:
            return
        try:
            self.status_page.update(stage=stage, headline=headline, detail=detail, **kwargs)
        except Exception as exc:
            self.logger.warning("Status page update failed: %s", exc)
            self.status_page = None
