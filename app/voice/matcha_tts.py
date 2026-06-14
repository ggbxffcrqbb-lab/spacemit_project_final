import glob
import importlib.util
import os
import re
import tempfile
import threading
import time


DEFAULT_MATCHA_ROOT = os.getenv(
    "SPACEMIT_MATCHA_ROOT",
    "/mnt/ssd/spacemit_project/third_party/model-zoo-tts",
)
DEFAULT_MODEL_DIR = os.path.expanduser(
    os.getenv("SPACEMIT_MATCHA_MODEL_DIR", "~/.cache/models/tts/matcha-tts")
)
DEFAULT_PROVIDER = os.getenv("SPACEMIT_MATCHA_PROVIDER", "cpu")
DEFAULT_PRESET = os.getenv("SPACEMIT_MATCHA_DEFAULT_PRESET", "matcha_zh")
MIXED_PRESET = os.getenv("SPACEMIT_MATCHA_MIXED_PRESET", "matcha_zh_en")
CHINESE_MODE = os.getenv("SPACEMIT_MATCHA_CHINESE_MODE", "quality").strip().lower()
DEFAULT_SPEED = float(os.getenv("SPACEMIT_MATCHA_SPEED", "1.0"))
DEFAULT_VOLUME = int(os.getenv("SPACEMIT_MATCHA_VOLUME", "100"))
DEFAULT_THREADS = int(os.getenv("SPACEMIT_MATCHA_THREADS", "4"))
ENABLE_NATIVE_WARMUP = os.getenv("SPACEMIT_MATCHA_ENABLE_WARMUP", "1") == "1"
ALLOW_CPU_FALLBACK = os.getenv("SPACEMIT_MATCHA_ALLOW_CPU_FALLBACK", "1") == "1"
PRELOAD_MIXED_ENGINE = os.getenv("SPACEMIT_MATCHA_PRELOAD_MIXED", "1") == "1"
WARMUP_TEXT_ZH = os.getenv("SPACEMIT_MATCHA_WARMUP_TEXT_ZH", "\u4f60\u597d\u3002")
WARMUP_TEXT_MIXED = os.getenv("SPACEMIT_MATCHA_WARMUP_TEXT_MIXED", "\u4eca\u5929\u5b66Python\u3002")


class TTSModel:
    _native_module = None
    _native_lock = threading.Lock()

    def __init__(self):
        self._native = self._load_native_module()
        self._matcha_root = DEFAULT_MATCHA_ROOT
        self._model_dir = DEFAULT_MODEL_DIR
        self._provider = DEFAULT_PROVIDER
        self._default_preset = DEFAULT_PRESET
        self._mixed_preset = MIXED_PRESET
        self._chinese_mode = CHINESE_MODE if CHINESE_MODE in {"quality", "fast"} else "quality"
        self._speed = DEFAULT_SPEED
        self._volume = DEFAULT_VOLUME
        self._threads = DEFAULT_THREADS
        self._engines = {}
        self._engine_meta = {}
        self._engine_lock = threading.RLock()
        self._prewarm_inflight = set()
        self._preload_started = False
        self._preload_thread = None
        self._last_metrics = {}

        # Fail fast so startup surfaces missing official resources immediately.
        self._get_engine(self._get_primary_preset())
        self._start_background_preload_if_needed()

    @classmethod
    def _load_native_module(cls):
        with cls._native_lock:
            if cls._native_module is not None:
                return cls._native_module

            so_candidates = glob.glob(
                os.path.join(
                    DEFAULT_MATCHA_ROOT,
                    "build",
                    "python",
                    "_spacemit_tts*.so",
                )
            )
            if not so_candidates:
                raise FileNotFoundError(
                    "Official Matcha native module not found. "
                    f"Expected under: {DEFAULT_MATCHA_ROOT}/build/python"
                )

            module_path = so_candidates[0]
            spec = importlib.util.spec_from_file_location(
                "_spacemit_tts",
                module_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load native module from {module_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls._native_module = module
            return module

    def _has_latin(self, text):
        return bool(re.search(r"[A-Za-z]", text))

    def _get_primary_preset(self):
        if self._chinese_mode == "fast":
            return self._mixed_preset
        return self._default_preset

    def _get_resident_presets(self):
        presets = [self._get_primary_preset()]
        if PRELOAD_MIXED_ENGINE and self._mixed_preset not in presets:
            presets.append(self._mixed_preset)
        return presets

    def _select_preset(self, text):
        if self._chinese_mode == "fast":
            return self._mixed_preset
        return self._mixed_preset if self._has_latin(text) else self._default_preset

    def _build_config(self, preset_name, provider):
        config = self._native.TtsConfig.preset(preset_name)
        config.model_dir = self._model_dir
        config.provider = provider
        config.speech_rate = self._speed
        config.volume = self._volume
        config.num_threads = self._threads
        config.enable_warmup = ENABLE_NATIVE_WARMUP
        return config

    def _create_engine(self, preset_name, provider):
        init_begin = time.perf_counter()
        config = self._build_config(preset_name, provider)
        engine = self._native.TtsEngine(config)
        init_wall_ms = round((time.perf_counter() - init_begin) * 1000.0, 3)
        if not engine.is_initialized():
            raise RuntimeError(
                f"Failed to initialize Matcha engine: preset={preset_name}, provider={provider}"
            )
        self._engine_meta[preset_name] = {
            "preset": preset_name,
            "provider": provider,
            "init_wall_ms": init_wall_ms,
            "native_warmup": ENABLE_NATIVE_WARMUP,
            "sample_rate": engine.get_sample_rate(),
            "prewarm_wall_ms": None,
            "prewarm_proc_ms": None,
            "prewarm_rtf": None,
        }
        print(
            f"[MATCHA INIT] preset={preset_name} provider={provider} "
            f"threads={self._threads} warmup={int(ENABLE_NATIVE_WARMUP)} "
            f"init={init_wall_ms}ms sample_rate={engine.get_sample_rate()}Hz"
        )
        return engine

    def _get_engine(self, preset_name):
        with self._engine_lock:
            if preset_name in self._engines:
                return self._engines[preset_name]

            provider = self._provider
            try:
                engine = self._create_engine(preset_name, provider)
                active_provider = provider
            except Exception as exc:
                if provider != "cpu" and ALLOW_CPU_FALLBACK:
                    print(
                        f"[MATCHA WARN] preset={preset_name} provider={provider} failed: {exc}"
                    )
                    print(f"[MATCHA WARN] fallback to provider=cpu for preset={preset_name}")
                    engine = self._create_engine(preset_name, "cpu")
                    active_provider = "cpu"
                else:
                    raise

            self._engines[preset_name] = (engine, active_provider)
            self._engine_meta[preset_name]["provider"] = active_provider
            return self._engines[preset_name]

    def _warmup_text_for_preset(self, preset_name):
        if preset_name == self._mixed_preset:
            return WARMUP_TEXT_MIXED
        return WARMUP_TEXT_ZH

    def _prewarm_preset(self, preset_name):
        while True:
            with self._engine_lock:
                meta = self._engine_meta.get(preset_name)
                if meta and meta.get("prewarm_wall_ms") is not None:
                    return
                if preset_name not in self._prewarm_inflight:
                    self._prewarm_inflight.add(preset_name)
                    break
            time.sleep(0.05)

        engine, provider = self._get_engine(preset_name)
        try:
            warmup_text = self._warmup_text_for_preset(preset_name)
            warmup_begin = time.perf_counter()
            result = engine.call(warmup_text)
            warmup_wall_ms = round((time.perf_counter() - warmup_begin) * 1000.0, 3)
            if not result.is_success():
                raise RuntimeError(
                    f"Warmup synthesis failed: preset={preset_name}, provider={provider}, "
                    f"message={result.get_message()}"
                )
            meta = self._engine_meta.setdefault(preset_name, {})
            meta["provider"] = provider
            meta["prewarm_wall_ms"] = warmup_wall_ms
            meta["prewarm_proc_ms"] = result.get_processing_time_ms()
            meta["prewarm_rtf"] = result.get_rtf()
            print(
                f"[MATCHA PREWARM] preset={preset_name} provider={provider} "
                f"wall={warmup_wall_ms}ms proc={result.get_processing_time_ms()}ms "
                f"rtf={result.get_rtf():.3f}"
            )
        finally:
            with self._engine_lock:
                self._prewarm_inflight.discard(preset_name)

    def _background_preload(self):
        try:
            for preset_name in self._get_resident_presets():
                if preset_name == self._get_primary_preset():
                    continue
                self._prewarm_preset(preset_name)
        except Exception as exc:
            print(f"[MATCHA WARN] background preload failed: {exc}")

    def _start_background_preload_if_needed(self):
        if not PRELOAD_MIXED_ENGINE:
            return
        if self._mixed_preset == self._get_primary_preset():
            return
        with self._engine_lock:
            if self._preload_started:
                return
            self._preload_started = True
            self._preload_thread = threading.Thread(
                target=self._background_preload,
                name="matcha-preload",
                daemon=True,
            )
            self._preload_thread.start()

    def prewarm(self):
        for preset_name in self._get_resident_presets():
            self._prewarm_preset(preset_name)

    def get_engine_status(self):
        with self._engine_lock:
            return {
                preset: dict(meta)
                for preset, meta in self._engine_meta.items()
            }

    def get_last_metrics(self):
        return dict(self._last_metrics)

    def ort_predict(self, text, queue_wait_ms=None):
        preset_name = self._select_preset(text)
        engine, provider = self._get_engine(preset_name)
        synth_begin = time.perf_counter()
        result = engine.call(text)
        synth_wall_ms = round((time.perf_counter() - synth_begin) * 1000.0, 3)
        if not result.is_success():
            raise RuntimeError(result.get_message())

        fd, wav_path = tempfile.mkstemp(prefix="matcha_", suffix=".wav")
        os.close(fd)
        if not result.save_to_file(wav_path):
            raise RuntimeError(f"Failed to save Matcha output to {wav_path}")

        self._last_metrics = {
            "preset": preset_name,
            "provider": provider,
            "route_mode": self._chinese_mode,
            "duration_ms": result.get_duration_ms(),
            "processing_time_ms": result.get_processing_time_ms(),
            "synth_wall_ms": synth_wall_ms,
            "rtf": result.get_rtf(),
            "sample_rate": result.get_sample_rate(),
            "queue_wait_ms": queue_wait_ms,
            "init_wall_ms": self._engine_meta.get(preset_name, {}).get("init_wall_ms"),
            "prewarm_wall_ms": self._engine_meta.get(preset_name, {}).get("prewarm_wall_ms"),
        }
        print(
            f"[MATCHA] preset={preset_name} provider={provider} "
            f"duration={result.get_duration_ms()}ms proc={result.get_processing_time_ms()}ms "
            f"wall={synth_wall_ms}ms queue_wait={queue_wait_ms}ms "
            f"rtf={result.get_rtf():.3f}"
        )
        return wav_path
