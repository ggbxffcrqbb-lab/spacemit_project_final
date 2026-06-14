from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import RagConfig
from app.rag import LocalKnowledgeBase


def main():
    config = RagConfig(
        enabled=True,
        knowledge_dir=PROJECT_ROOT / "data" / "knowledge",
        index_path=PROJECT_ROOT / "data" / "index" / "test_knowledge_index.json",
        top_k=3,
        chunk_max_chars=220,
        min_score=0.8,
        max_context_chars=900,
        citation_limit=2,
        direct_answer_score=2.0,
    )
    kb = LocalKnowledgeBase(config)
    hits = kb.search("涂层起泡怎么办")
    print(kb.get_status())
    for hit in hits:
        print(hit.score, hit.source_label)


if __name__ == "__main__":
    main()
