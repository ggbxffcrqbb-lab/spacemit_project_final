from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import load_app_config
from app.core.logging_utils import setup_logging
from app.rag import LocalKnowledgeBase
from app.rag.importer import KnowledgeImporter


MODE_TO_CONFIG = {
    "default": Path("configs/voice.yaml"),
    "fast": Path("configs/voice_fast.yaml"),
    "multimodal_demo": Path("configs/multimodal_demo.yaml"),
    "voice_guided_demo": Path("configs/voice_guided_demo.yaml"),
    "vision": Path("configs/vision.yaml"),
    "vision_usb": Path("configs/vision_usb.yaml"),
    "vision_usb_demo": Path("configs/vision_usb_demo.yaml"),
    "vision_usb_live": Path("configs/vision_usb_live.yaml"),
    "vision_usb_corrosion_rt": Path("configs/vision_usb_corrosion_two_stage_rt.yaml"),
}


def build_parser():
    parser = argparse.ArgumentParser(description="Spacemit board-side application entry")
    parser.add_argument(
        "--config",
        help="Path to the project config file. Overrides --mode when provided.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_CONFIG.keys()),
        default="default",
        help="Built-in runtime mode preset",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("voice-console", help="Run the resident voice console")
    subparsers.add_parser("warmup", help="Warm up resident ASR/LLM/TTS runtimes")
    subparsers.add_parser("doctor", help="Print current runtime health report")

    multimodal_demo = subparsers.add_parser(
        "multimodal-demo",
        help="Run the Phase 6 multimodal terminal dashboard and competition demo",
    )
    multimodal_demo.add_argument("--backend", default="", help="Override camera backend")
    multimodal_demo.add_argument("--recognizer", default="", help="Override recognizer backend")
    multimodal_demo.add_argument(
        "--interval-seconds",
        type=float,
        default=0.0,
        help="Target delay between analysis rounds. Use 0.0 for the fastest live loop.",
    )
    multimodal_demo.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for scripted tests. Use 0 to keep running.",
    )
    multimodal_demo.add_argument(
        "--headless-vision",
        action="store_true",
        help="Disable the fullscreen competition display and keep only the terminal dashboard.",
    )

    voice_guided_demo = subparsers.add_parser(
        "voice-guided-demo",
        help="Run the voice-guided camera selection and inspection demo",
    )
    voice_guided_demo.add_argument(
        "--headless-vision",
        action="store_true",
        help="Disable the fullscreen competition display and keep only the terminal dashboard.",
    )

    rag_query = subparsers.add_parser("rag-query", help="Run a local RAG search")
    rag_query.add_argument("text", help="Question text for retrieval")
    rag_query.add_argument("--top-k", type=int, default=3, help="Number of retrieval hits")

    subparsers.add_parser("rag-rebuild", help="Rebuild the local RAG index cache")

    knowledge_import = subparsers.add_parser(
        "knowledge-import",
        help="Import markdown/text/pdf/docx files into the local knowledge base",
    )
    knowledge_import.add_argument("sources", nargs="+", help="Files or directories to import")
    knowledge_import.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories",
    )
    knowledge_import.add_argument(
        "--category",
        default="",
        help="Optional category label added to imported documents",
    )
    knowledge_import.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional tag. Can be repeated.",
    )
    knowledge_import.add_argument(
        "--title-prefix",
        default="",
        help="Optional title prefix for imported files",
    )
    knowledge_import.add_argument(
        "--dest-subdir",
        default="imported",
        help="Relative subdirectory under data/knowledge",
    )
    knowledge_import.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite by title slug instead of hash-suffixed filenames",
    )
    knowledge_import.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip automatic RAG index rebuild after import",
    )

    vision_doctor = subparsers.add_parser("vision-doctor", help="Print current vision pipeline report")
    vision_doctor.add_argument("--backend", default="", help="Override camera backend")
    vision_doctor.add_argument(
        "--probe",
        action="store_true",
        help="Run deep camera probing instead of only static dependency checks",
    )

    vision_capture = subparsers.add_parser(
        "vision-capture-once",
        help="Capture a single frame from the selected camera backend",
    )
    vision_capture.add_argument("--backend", default="", help="Override camera backend")
    vision_capture.add_argument("--output", default="", help="Optional saved image path")

    vision_preview = subparsers.add_parser(
        "vision-preview",
        help="Launch a backend-specific live camera preview",
    )
    vision_preview.add_argument("--backend", default="", help="Override camera backend")
    vision_preview.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the preview command without launching it",
    )

    vision_image = subparsers.add_parser(
        "vision-image",
        help="Analyze an existing image with the selected recognizer backend",
    )
    vision_image.add_argument("image", help="Path to the input image")
    vision_image.add_argument("--recognizer", default="", help="Override recognizer backend")
    vision_image.add_argument("--annotated-output", default="", help="Optional annotated image path")
    vision_image.add_argument("--result-json", default="", help="Optional saved analysis json path")

    vision_camera = subparsers.add_parser(
        "vision-camera",
        help="Capture one frame and run the vision recognizer on it",
    )
    vision_camera.add_argument("--backend", default="", help="Override camera backend")
    vision_camera.add_argument("--recognizer", default="", help="Override recognizer backend")
    vision_camera.add_argument("--output", default="", help="Optional captured image path")
    vision_camera.add_argument("--annotated-output", default="", help="Optional annotated image path")
    vision_camera.add_argument("--result-json", default="", help="Optional saved analysis json path")

    vision_stream = subparsers.add_parser(
        "vision-stream",
        help="Continuously capture frames and update a vision status page",
    )
    vision_stream.add_argument("--backend", default="", help="Override camera backend")
    vision_stream.add_argument("--recognizer", default="", help="Override recognizer backend")
    vision_stream.add_argument(
        "--interval-seconds",
        type=float,
        default=1.5,
        help="Delay target between frames. Higher values reduce load and improve stability.",
    )
    vision_stream.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. Use 0 to keep running until interrupted.",
    )
    vision_stream.add_argument(
        "--display-status-page",
        action="store_true",
        help="Deprecated compatibility flag. Now only enables the native preview window and does not open the browser status page.",
    )
    vision_stream.add_argument(
        "--display-preview",
        action="store_true",
        help="Enable the native preview window during streaming without using the browser status page.",
    )
    vision_stream.add_argument(
        "--display-competition",
        action="store_true",
        help="Show a native fullscreen competition display with live camera video and overlayed result text.",
    )
    vision_stream.add_argument(
        "--performance-mode",
        action="store_true",
        default=None,
        help="Disable per-frame disk writes and status-page generation for lower latency.",
    )
    return parser


def resolve_config_path(args) -> Path:
    if args.config:
        return Path(args.config).expanduser()
    return MODE_TO_CONFIG[args.mode]


def _print_json(data: dict, output_path: str | None = None) -> None:
    rendered = json.dumps(data, ensure_ascii=False, indent=2)
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _run_rag_commands(config, args) -> bool:
    if args.command not in {"rag-query", "rag-rebuild", "knowledge-import"}:
        return False

    if args.command == "knowledge-import":
        importer = KnowledgeImporter(config.rag)
        result = importer.import_sources(
            sources=args.sources,
            recursive=args.recursive,
            category=args.category,
            tags=args.tag,
            title_prefix=args.title_prefix,
            dest_subdir=args.dest_subdir,
            rebuild=not args.no_rebuild,
            replace=args.replace,
        )
        _print_json(result)
        return True

    kb = LocalKnowledgeBase(config.rag)
    if args.command == "rag-query":
        hits = kb.search(args.text, top_k=args.top_k)
        _print_json(
            {
                "query": args.text,
                "rag_status": kb.get_status(),
                "hits": [
                    {
                        "score": hit.score,
                        "title": hit.title,
                        "source_path": hit.source_path,
                        "source_label": hit.source_label,
                        "overlap_terms": hit.overlap_terms,
                        "text": hit.text,
                    }
                    for hit in hits
                ],
            }
        )
    else:
        kb.rebuild()
        _print_json(kb.get_status())
    return True


def _run_vision_commands(config, args) -> bool:
    if args.command not in {
        "vision-doctor",
        "vision-capture-once",
        "vision-preview",
        "vision-image",
        "vision-camera",
        "vision-stream",
    }:
        return False

    from app.vision import VisionPipelineService

    service = VisionPipelineService(config)
    if args.command == "vision-doctor":
        _print_json(service.doctor(args.backend or None, args.probe))
    elif args.command == "vision-capture-once":
        _print_json(service.capture_once(args.backend or None, args.output or None))
    elif args.command == "vision-preview":
        _print_json(service.preview_camera(args.backend or None, args.dry_run))
    elif args.command == "vision-image":
        _print_json(
            service.analyze_image(
                args.image,
                args.recognizer or None,
                args.annotated_output or None,
            ),
            args.result_json or None,
        )
    elif args.command == "vision-stream":
        _print_json(
            service.stream_camera(
                args.backend or None,
                args.recognizer or None,
                args.interval_seconds,
                args.max_frames,
                args.display_status_page or args.display_preview,
                args.performance_mode,
                args.display_competition,
            )
        )
    else:
        _print_json(
            service.analyze_camera(
                args.backend or None,
                args.recognizer or None,
                args.output or None,
                args.annotated_output or None,
            ),
            args.result_json or None,
        )
    return True


def _run_multimodal_commands(config, args) -> bool:
    if args.command not in {"multimodal-demo", "voice-guided-demo"}:
        return False

    if args.command == "voice-guided-demo":
        from app.core.voice_guided_demo_controller import VoiceGuidedDemoController

        controller = VoiceGuidedDemoController(
            config,
            display_competition=not args.headless_vision,
            performance_mode=True,
        )
    else:
        from app.core.multimodal_controller import MultimodalDemoController

        controller = MultimodalDemoController(
            config,
            camera_backend=args.backend or None,
            recognizer_backend=args.recognizer or None,
            interval_seconds=args.interval_seconds,
            max_frames=args.max_frames,
            display_competition=not args.headless_vision,
            performance_mode=True,
        )
    _print_json(controller.run())
    return True


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_app_config(resolve_config_path(args))
    setup_logging(
        log_dir=config.logging.dir,
        level=config.logging.level,
        runtime_file=config.logging.runtime_file,
    )

    if _run_rag_commands(config, args):
        return

    if _run_multimodal_commands(config, args):
        return

    if _run_vision_commands(config, args):
        return

    from app.voice.service import ResidentVoiceService

    service = ResidentVoiceService(config)
    try:
        if args.command == "voice-console":
            service.run_console()
        elif args.command == "warmup":
            service.start_workers()
            service.warmup()
            _print_json(service.build_health_report())
        elif args.command == "doctor":
            _print_json(service.build_health_report())
        else:
            parser.error(f"Unsupported command: {args.command}")
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
