from __future__ import annotations

import audioop
import os
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import sounddevice as sd


def split_tts_segments(buffer: str, max_chars: int, min_chars: int, flush: bool = False):
    hard_endings = "。！？!?;\n；"
    soft_endings = "，,"
    segments = []
    start = 0

    for idx, char in enumerate(buffer):
        current = buffer[start : idx + 1].strip()
        if not current:
            continue

        if char in hard_endings:
            segments.append(current)
            start = idx + 1
            continue

        if char in soft_endings and len(current) >= max_chars:
            segments.append(current)
            start = idx + 1

    remaining = buffer[start:]
    if flush and remaining.strip():
        segments.append(remaining.strip())
        remaining = ""

    if not flush and remaining.strip() and len(remaining.strip()) >= max_chars:
        candidate = remaining.strip()
        split_at = max(candidate.rfind(sep) for sep in soft_endings + " ")
        if split_at >= min_chars:
            segments.append(candidate[: split_at + 1].strip())
            remaining = candidate[split_at + 1 :]

    return segments, remaining


class ConsoleAudioRecorder:
    def __init__(self, sample_rate: int = 48000, channels: int = 1, dtype: str = "int16"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

    def record_once(self, auto_start: bool = False) -> str:
        if not auto_start:
            print("Press Enter to start recording, q and Enter to exit.")
            s = input()
            if s == "q":
                return ""
        else:
            print("Recording started. Press Enter to stop, q and Enter to exit.")

        if not auto_start:
            print("Recording... Press Enter again to stop, q and Enter to exit.")
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav_file.name
        frames = []

        def callback(indata, frame_count, time_info, status):
            if status:
                print(f"Recording status: {status}", flush=True)
            frames.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=callback,
        ):
            s = input()
            if s == "q":
                return ""

        with wave.open(temp_wav_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))

        print(f"Recording saved to {temp_wav_path}")
        return temp_wav_path


class AudioPlayer:
    def __init__(self, device: str, volume: str, sample_rate: int, channels: int):
        self.device = device
        self.volume = volume
        self.sample_rate = sample_rate
        self.channels = channels
        self._volume_ready = False
        self._volume_lock = threading.Lock()

    def _prepare_audio_for_playback(self, audio_file: str):
        with wave.open(audio_file, "rb") as rf:
            channels = rf.getnchannels()
            sample_width = rf.getsampwidth()
            sample_rate = rf.getframerate()
            data = rf.readframes(rf.getnframes())

        converted = False
        if channels == 1 and self.channels == 2:
            data = audioop.tostereo(data, sample_width, 1, 1)
            channels = 2
            converted = True

        if sample_rate != self.sample_rate:
            data, _ = audioop.ratecv(
                data,
                sample_width,
                channels,
                sample_rate,
                self.sample_rate,
                None,
            )
            sample_rate = self.sample_rate
            converted = True

        if not converted:
            return audio_file, False

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        with wave.open(temp_wav.name, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(data)
        return temp_wav.name, True

    def _ensure_volume(self):
        with self._volume_lock:
            if self._volume_ready:
                return
            card = self.device.split(":")[1].split(",")[0]
            subprocess.run(
                ["amixer", "-c", card, "set", "PCM", self.volume],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._volume_ready = True

    def play_audio(self, audio_file: str):
        playback_file = audio_file
        playback_is_temp = False
        try:
            if not Path(audio_file).exists():
                raise FileNotFoundError(audio_file)

            self._ensure_volume()
            playback_file, playback_is_temp = self._prepare_audio_for_playback(audio_file)
            proc = subprocess.run(
                ["aplay", "-D", self.device, playback_file],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"aplay failed rc={proc.returncode}, stderr={proc.stderr.strip()}"
                )
        finally:
            if playback_is_temp and playback_file and os.path.exists(playback_file):
                try:
                    os.remove(playback_file)
                except OSError:
                    pass
