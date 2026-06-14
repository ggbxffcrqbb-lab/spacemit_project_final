from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import RagConfig
from app.rag import KnowledgeImporter


def main():
    with tempfile.TemporaryDirectory(prefix="spacemit-import-smoke-") as tmp_dir:
        root = Path(tmp_dir)
        source_dir = root / "source_docs"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "sample.txt").write_text(
            "防腐巡检速记\n发现涂层起泡时，先确认是否伴随锈蚀和渗漏，再安排复核。",
            encoding="utf-8",
        )
        (source_dir / "guide.md").write_text(
            "# 储罐外壁检查\n\n先看焊缝、支撑处和积水阴影区，再判断是否需要补测。",
            encoding="utf-8",
        )

        knowledge_dir = root / "knowledge"
        config = RagConfig(
            enabled=True,
            knowledge_dir=knowledge_dir,
            index_path=root / "index" / "knowledge_index.json",
            top_k=3,
            chunk_max_chars=220,
            min_score=0.8,
            max_context_chars=900,
            citation_limit=2,
            direct_answer_score=2.0,
        )
        importer = KnowledgeImporter(config)
        result = importer.import_sources(
            sources=[str(source_dir)],
            recursive=True,
            category="smoke",
            tags=["board", "import"],
            title_prefix="验证-",
            dest_subdir="imported_smoke",
            rebuild=True,
        )
        print(result)


if __name__ == "__main__":
    main()
