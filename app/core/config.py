from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectPaths:
    root_dir: Path
    models_dir: Path
    data_dir: Path
    cache_dir: Path


@dataclass
class LoggingConfig:
    level: str
    dir: Path
    runtime_file: str
    turns_file: str

    @property
    def runtime_path(self) -> Path:
        return self.dir / self.runtime_file

    @property
    def turns_path(self) -> Path:
        return self.dir / self.turns_file


@dataclass
class AsrConfig:
    model_dir: Path
    prefer_optimized_model: bool
    language: str
    use_itn: bool
    batch_size: int
    intra_op_num_threads: int


@dataclass
class LlmConfig:
    model: str
    api_url: str
    system_prompt: str
    max_tokens: int
    temperature: float
    num_ctx: int
    keep_alive: str
    timeout: int
    max_chars: int
    min_chars: int
    stop_after_first_sentence: bool
    warmup_prompt: str


@dataclass
class TtsConfig:
    matcha_root: Path
    model_dir: Path
    provider: str
    default_preset: str
    mixed_preset: str
    chinese_mode: str
    speed: float
    volume: int
    threads: int
    enable_warmup: bool
    allow_cpu_fallback: bool
    preload_mixed_engine: bool
    warmup_text_zh: str
    warmup_text_mixed: str
    keep_tts_wav: bool
    trace_init: bool


@dataclass
class RagConfig:
    enabled: bool
    knowledge_dir: Path
    index_path: Path
    top_k: int
    chunk_max_chars: int
    min_score: float
    max_context_chars: int
    citation_limit: int
    direct_answer_score: float


@dataclass
class UiConfig:
    enabled: bool
    status_dir: Path
    html_file: str
    json_file: str
    text_file: str
    title: str
    refresh_seconds: int
    history_limit: int

    @property
    def html_path(self) -> Path:
        return self.status_dir / self.html_file

    @property
    def json_path(self) -> Path:
        return self.status_dir / self.json_file

    @property
    def text_path(self) -> Path:
        return self.status_dir / self.text_file


@dataclass
class VoiceConfig:
    playback_device: str
    playback_volume: str
    playback_sample_rate: int
    playback_channels: int
    segment_max_chars: int
    segment_min_chars: int
    prewarm_llm: bool
    prewarm_tts: bool
    asr: AsrConfig
    llm: LlmConfig
    tts: TtsConfig


@dataclass
class AppConfig:
    project_name: str
    paths: ProjectPaths
    logging: LoggingConfig
    voice: VoiceConfig
    rag: RagConfig
    ui: UiConfig
    config_path: Path


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_app_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    project = raw["project"]
    logging_raw = raw["logging"]
    voice_raw = raw["voice"]
    asr_raw = voice_raw["asr"]
    llm_raw = voice_raw["llm"]
    tts_raw = voice_raw["tts"]
    rag_raw = raw.get("rag", {})
    ui_raw = raw.get("ui", {})

    project_paths = ProjectPaths(
        root_dir=_path(project["root_dir"]),
        models_dir=_path(project["models_dir"]),
        data_dir=_path(project["data_dir"]),
        cache_dir=_path(project["cache_dir"]),
    )
    default_ui_dir = _path(logging_raw["dir"]).parent / "ui"

    return AppConfig(
        project_name=project["name"],
        paths=project_paths,
        logging=LoggingConfig(
            level=logging_raw["level"],
            dir=_path(logging_raw["dir"]),
            runtime_file=logging_raw["runtime_file"],
            turns_file=logging_raw["turns_file"],
        ),
        voice=VoiceConfig(
            playback_device=voice_raw["playback_device"],
            playback_volume=voice_raw["playback_volume"],
            playback_sample_rate=int(voice_raw["playback_sample_rate"]),
            playback_channels=int(voice_raw["playback_channels"]),
            segment_max_chars=int(voice_raw["segment_max_chars"]),
            segment_min_chars=int(voice_raw["segment_min_chars"]),
            prewarm_llm=_bool(voice_raw["prewarm_llm"]),
            prewarm_tts=_bool(voice_raw["prewarm_tts"]),
            asr=AsrConfig(
                model_dir=_path(asr_raw["model_dir"]),
                prefer_optimized_model=_bool(asr_raw.get("prefer_optimized_model", True)),
                language=asr_raw["language"],
                use_itn=_bool(asr_raw["use_itn"]),
                batch_size=int(asr_raw["batch_size"]),
                intra_op_num_threads=int(asr_raw["intra_op_num_threads"]),
            ),
            llm=LlmConfig(
                model=llm_raw["model"],
                api_url=llm_raw["api_url"],
                system_prompt=llm_raw["system_prompt"],
                max_tokens=int(llm_raw["max_tokens"]),
                temperature=float(llm_raw["temperature"]),
                num_ctx=int(llm_raw["num_ctx"]),
                keep_alive=llm_raw["keep_alive"],
                timeout=int(llm_raw["timeout"]),
                max_chars=int(llm_raw["max_chars"]),
                min_chars=int(llm_raw["min_chars"]),
                stop_after_first_sentence=_bool(llm_raw["stop_after_first_sentence"]),
                warmup_prompt=llm_raw["warmup_prompt"],
            ),
            tts=TtsConfig(
                matcha_root=_path(tts_raw["matcha_root"]),
                model_dir=_path(tts_raw["model_dir"]),
                provider=tts_raw["provider"],
                default_preset=tts_raw["default_preset"],
                mixed_preset=tts_raw["mixed_preset"],
                chinese_mode=tts_raw["chinese_mode"],
                speed=float(tts_raw["speed"]),
                volume=int(tts_raw["volume"]),
                threads=int(tts_raw["threads"]),
                enable_warmup=_bool(tts_raw["enable_warmup"]),
                allow_cpu_fallback=_bool(tts_raw["allow_cpu_fallback"]),
                preload_mixed_engine=_bool(tts_raw["preload_mixed_engine"]),
                warmup_text_zh=tts_raw["warmup_text_zh"],
                warmup_text_mixed=tts_raw["warmup_text_mixed"],
                keep_tts_wav=_bool(tts_raw["keep_tts_wav"]),
                trace_init=_bool(tts_raw["trace_init"]),
            ),
        ),
        rag=RagConfig(
            enabled=_bool(rag_raw.get("enabled", False)),
            knowledge_dir=_path(
                rag_raw.get("knowledge_dir", str(project_paths.root_dir / "data" / "knowledge"))
            ),
            index_path=_path(
                rag_raw.get("index_path", str(project_paths.root_dir / "data" / "index" / "knowledge_index.json"))
            ),
            top_k=int(rag_raw.get("top_k", 3)),
            chunk_max_chars=int(rag_raw.get("chunk_max_chars", 220)),
            min_score=float(rag_raw.get("min_score", 0.8)),
            max_context_chars=int(rag_raw.get("max_context_chars", 900)),
            citation_limit=int(rag_raw.get("citation_limit", 2)),
            direct_answer_score=float(rag_raw.get("direct_answer_score", 2.0)),
        ),
        ui=UiConfig(
            enabled=_bool(ui_raw.get("enabled", True)),
            status_dir=_path(
                ui_raw.get("status_dir", str(default_ui_dir))
            ),
            html_file=ui_raw.get("html_file", "status_page.html"),
            json_file=ui_raw.get("json_file", "status_state.json"),
            text_file=ui_raw.get("text_file", "status_screen.txt"),
            title=ui_raw.get("title", "防腐专家状态页"),
            refresh_seconds=int(ui_raw.get("refresh_seconds", 3)),
            history_limit=int(ui_raw.get("history_limit", 6)),
        ),
        config_path=config_path,
    )
