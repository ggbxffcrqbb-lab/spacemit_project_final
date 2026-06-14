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
    return parser


def resolve_config_path(args) -> Path:
    if args.config:
        return Path(args.config).expanduser()
    return MODE_TO_CONFIG[args.mode]


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_app_config(resolve_config_path(args))
    setup_logging(
        log_dir=config.logging.dir,
        level=config.logging.level,
        runtime_file=config.logging.runtime_file,
    )

    if args.command in {"rag-query", "rag-rebuild", "knowledge-import"}:
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
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        kb = LocalKnowledgeBase(config.rag)
        if args.command == "rag-query":
            hits = kb.search(args.text, top_k=args.top_k)
            print(
                json.dumps(
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
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            kb.rebuild()
            print(json.dumps(kb.get_status(), ensure_ascii=False, indent=2))
        return

    from app.voice.service import ResidentVoiceService

    service = ResidentVoiceService(config)
    try:
        if args.command == "voice-console":
            service.run_console()
        elif args.command == "warmup":
            service.start_workers()
            service.warmup()
            print(json.dumps(service.build_health_report(), ensure_ascii=False, indent=2))
        elif args.command == "doctor":
            print(json.dumps(service.build_health_report(), ensure_ascii=False, indent=2))
        else:
            parser.error(f"Unsupported command: {args.command}")
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
