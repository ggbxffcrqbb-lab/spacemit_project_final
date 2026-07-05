from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
from sounddevice import PortAudioError


@dataclass
class PassiveUtterance:
    audio_path: Path
    duration_seconds: float
    rms: float
    created_at: float


class PassiveAudioListener:
    """Continuously capture microphone audio and emit VAD-bounded utterances."""

    def __init__(
        self,
        on_utterance,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        block_duration_ms: int = 100,
        energy_threshold: float = 0.012,
        start_speech_blocks: int = 2,
        end_silence_blocks: int = 6,
        min_speech_blocks: int = 3,
        max_utterance_seconds: float = 8.0,
        tail_padding_blocks: int = 2,
        queue_blocks: int = 128,
        latency_seconds: float | None = 0.20,
        never_drop_input: bool = True,
        reset_on_overflow: bool = True,
        device=None,
    ):
        self.on_utterance = on_utterance
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.block_duration_ms = int(block_duration_ms)
        self.energy_threshold = float(energy_threshold)
        self.start_speech_blocks = max(1, int(start_speech_blocks))
        self.end_silence_blocks = max(1, int(end_silence_blocks))
        self.min_speech_blocks = max(1, int(min_speech_blocks))
        self.max_utterance_seconds = max(1.0, float(max_utterance_seconds))
        self.tail_padding_blocks = max(0, int(tail_padding_blocks))
        self.queue_blocks = max(16, int(queue_blocks))
        self.latency_seconds = None if latency_seconds is None else max(0.05, float(latency_seconds))
        self.never_drop_input = bool(never_drop_input)
        self.reset_on_overflow = bool(reset_on_overflow)
        self.device = device
        self.logger = logging.getLogger("voice.passive_listener")

        self.blocksize = max(1, int(self.sample_rate * self.block_duration_ms / 1000))
        self._audio_queue: queue.Queue = queue.Queue(maxsize=self.queue_blocks)
        self._running = False
        self._paused = False
        self._pause_until = 0.0
        self._stream = None
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._overflow_flag = False

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="passive-audio-listener",
                daemon=True,
            )
            self._worker_thread.start()
            self._stream = self._open_input_stream()
            self._stream.start()
            self.logger.info(
                "Passive audio listener started sample_rate=%s channels=%s blocksize=%s queue_blocks=%s latency=%s never_drop_input=%s device=%s",
                self.sample_rate,
                self.channels,
                self.blocksize,
                self.queue_blocks,
                self.latency_seconds,
                self.never_drop_input,
                self.device,
            )

    def _open_input_stream(self):
        stream_kwargs = {
            "samplerate": self.sample_rate,
            "channels": self.channels,
            "dtype": "int16",
            "blocksize": self.blocksize,
            "latency": self.latency_seconds,
            "device": self.device,
            "callback": self._audio_callback,
        }
        attempts = []
        if self.never_drop_input:
            attempts.append({"never_drop_input": True})
        attempts.append({})

        last_error: Exception | None = None
        for extra_kwargs in attempts:
            try:
                return sd.RawInputStream(**stream_kwargs, **extra_kwargs)
            except PortAudioError as exc:
                last_error = exc
                if extra_kwargs.get("never_drop_input"):
                    self.logger.warning(
                        "Passive listener stream open failed with never_drop_input=True, falling back: %s",
                        exc,
                    )
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to open passive input stream")

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._paused = True
        self._audio_queue.put(None)
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self.logger.info("Passive audio listener stopped")

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._pause_until = 0.0

    def resume(self, delay_seconds: float = 0.0) -> None:
        with self._lock:
            self._paused = False
            self._pause_until = time.time() + max(0.0, float(delay_seconds))

    def configure_vad(
        self,
        *,
        block_duration_ms: int | None = None,
        end_silence_blocks: int | None = None,
        max_utterance_seconds: float | None = None,
        energy_threshold: float | None = None,
        start_speech_blocks: int | None = None,
        min_speech_blocks: int | None = None,
    ) -> None:
        with self._lock:
            if block_duration_ms is not None:
                block_duration_ms = max(40, int(block_duration_ms))
                if block_duration_ms != self.block_duration_ms:
                    self.block_duration_ms = block_duration_ms
                    self.blocksize = max(1, int(self.sample_rate * self.block_duration_ms / 1000))
                    self.logger.info(
                        "Passive listener updated block_duration_ms=%s blocksize=%s",
                        self.block_duration_ms,
                        self.blocksize,
                    )
            if end_silence_blocks is not None:
                self.end_silence_blocks = max(1, int(end_silence_blocks))
            if max_utterance_seconds is not None:
                self.max_utterance_seconds = max(1.0, float(max_utterance_seconds))
            if energy_threshold is not None:
                self.energy_threshold = max(0.001, float(energy_threshold))
            if start_speech_blocks is not None:
                self.start_speech_blocks = max(1, int(start_speech_blocks))
            if min_speech_blocks is not None:
                self.min_speech_blocks = max(1, int(min_speech_blocks))
            self.logger.info(
                (
                    "Passive listener VAD profile block_duration_ms=%s "
                    "end_silence_blocks=%s max_utterance_seconds=%.2f "
                    "energy_threshold=%.4f start_speech_blocks=%s min_speech_blocks=%s"
                ),
                self.block_duration_ms,
                self.end_silence_blocks,
                self.max_utterance_seconds,
                self.energy_threshold,
                self.start_speech_blocks,
                self.min_speech_blocks,
            )

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.logger.warning("Passive listener input status: %s", status)
            if self.reset_on_overflow and "overflow" in str(status).lower():
                with self._lock:
                    self._overflow_flag = True
        with self._lock:
            if not self._running:
                return
            paused = self._paused or time.time() < self._pause_until
        if paused:
            return
        payload = bytes(indata)
        try:
            self._audio_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(payload)
                self.logger.warning(
                    "Passive listener queue is full, dropped oldest audio block to keep latest capture"
                )
            except queue.Empty:
                self.logger.warning("Passive listener queue is full, dropping one audio block")

    def _worker_loop(self) -> None:
        pre_roll: list[bytes] = []
        active_chunks: list[bytes] = []
        speaking = False
        start_hits = 0
        silence_hits = 0
        speech_blocks = 0
        speech_peak_rms = 0.0
        speech_started_at = 0.0

        while True:
            item = self._audio_queue.get()
            if item is None:
                return

            with self._lock:
                paused = self._paused or time.time() < self._pause_until
                overflowed = self._overflow_flag
                self._overflow_flag = False
            if paused:
                pre_roll.clear()
                active_chunks.clear()
                speaking = False
                start_hits = 0
                silence_hits = 0
                speech_blocks = 0
                speech_peak_rms = 0.0
                speech_started_at = 0.0
                continue
            if overflowed:
                drained_blocks = self._drain_audio_queue()
                pre_roll.clear()
                active_chunks.clear()
                speaking = False
                start_hits = 0
                silence_hits = 0
                speech_blocks = 0
                speech_peak_rms = 0.0
                speech_started_at = 0.0
                self.logger.info(
                    "Passive listener reset VAD state after input overflow, drained_blocks=%s",
                    drained_blocks,
                )
                continue

            rms = self._compute_rms(item)
            is_speech = rms >= self.energy_threshold
            speech_peak_rms = max(speech_peak_rms, rms)

            if not speaking:
                pre_roll.append(item)
                if len(pre_roll) > self.tail_padding_blocks + self.start_speech_blocks:
                    pre_roll.pop(0)
                if is_speech:
                    start_hits += 1
                else:
                    start_hits = 0
                if start_hits >= self.start_speech_blocks:
                    speaking = True
                    active_chunks = list(pre_roll)
                    speech_blocks = len(active_chunks)
                    silence_hits = 0
                    speech_started_at = time.time()
                continue

            active_chunks.append(item)
            speech_blocks += 1
            if is_speech:
                silence_hits = 0
            else:
                silence_hits += 1

            utterance_seconds = (speech_blocks * self.block_duration_ms) / 1000.0
            if silence_hits < self.end_silence_blocks and utterance_seconds < self.max_utterance_seconds:
                continue

            if speech_blocks >= self.min_speech_blocks:
                self._emit_utterance(
                    active_chunks,
                    duration_seconds=utterance_seconds,
                    rms=speech_peak_rms,
                )

            pre_roll.clear()
            active_chunks = []
            speaking = False
            start_hits = 0
            silence_hits = 0
            speech_blocks = 0
            speech_peak_rms = 0.0
            speech_started_at = 0.0

    def _emit_utterance(self, chunks: list[bytes], *, duration_seconds: float, rms: float) -> None:
        if not chunks:
            return
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        path = Path(temp_wav.name)
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(chunks))
            utterance = PassiveUtterance(
                audio_path=path,
                duration_seconds=duration_seconds,
                rms=rms,
                created_at=time.time(),
            )
            self.on_utterance(utterance)
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def _drain_audio_queue(self) -> int:
        drained = 0
        while True:
            try:
                item = self._audio_queue.get_nowait()
            except queue.Empty:
                return drained
            if item is None:
                self._audio_queue.put_nowait(None)
                return drained
            drained += 1

    @staticmethod
    def _compute_rms(chunk: bytes) -> float:
        audio = np.frombuffer(chunk, dtype=np.int16)
        if audio.size == 0:
            return 0.0
        normalized = audio.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(normalized * normalized)))
