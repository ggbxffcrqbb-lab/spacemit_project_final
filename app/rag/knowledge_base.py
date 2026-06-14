from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import logging
from math import log
from pathlib import Path
import re
import time
from typing import Iterable

try:
    import jieba
except ModuleNotFoundError:  # pragma: no cover - fallback for thin local environments
    jieba = None

if jieba is not None:
    jieba.setLogLevel(logging.ERROR)


STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "对",
    "在",
    "要",
    "是",
    "后",
    "前",
    "时",
    "把",
    "将",
    "按",
    "用",
    "再",
    "先",
    "并",
    "做",
    "一个",
    "一种",
    "进行",
    "需要",
    "可以",
    "应当",
    "建议",
    "如果",
    "当前",
    "现场",
}

SECTION_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])")


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_path: str
    title: str
    text: str
    score: float
    overlap_terms: list[str]

    @property
    def source_label(self) -> str:
        return f"{self.title}（{Path(self.source_path).name}）"


@dataclass
class _KnowledgeChunk:
    chunk_id: str
    source_path: str
    title: str
    text: str
    tokens: list[str]
    token_counts: dict[str, int]
    length: int


class LocalKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self._chunks: list[_KnowledgeChunk] = []
        self._documents: list[str] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_doc_len = 0.0
        self._last_build_at = ""
        self._index_loaded_from_cache = False

        if self.config.enabled:
            self._load_or_build()

    def _load_or_build(self):
        if self._try_load_cache():
            return
        self.rebuild()

    def rebuild(self):
        if not self.config.knowledge_dir.exists():
            self.config.knowledge_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[_KnowledgeChunk] = []
        documents: list[str] = []
        for path in sorted(self._iter_knowledge_files()):
            documents.append(str(path))
            chunks.extend(self._build_chunks_for_file(path))

        self._chunks = chunks
        self._documents = documents
        self._doc_freq = Counter()
        total_length = 0
        for chunk in self._chunks:
            total_length += chunk.length
            for token in set(chunk.tokens):
                self._doc_freq[token] += 1

        self._avg_doc_len = total_length / len(self._chunks) if self._chunks else 0.0
        self._last_build_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._index_loaded_from_cache = False
        self._write_cache()

    def _iter_knowledge_files(self) -> Iterable[Path]:
        for path in self.config.knowledge_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            if path.name.startswith("."):
                continue
            if path.name.lower() == "readme.md":
                continue
            yield path

    def _build_chunks_for_file(self, path: Path) -> list[_KnowledgeChunk]:
        raw = path.read_text(encoding="utf-8")
        normalized = self._normalize_text(raw)
        doc_title = self._extract_doc_title(path, normalized)
        sections = self._split_sections(normalized, doc_title)

        chunks: list[_KnowledgeChunk] = []
        chunk_index = 0
        for section_title, section_text in sections:
            for chunk_text in self._chunk_text(section_text):
                tokens = self._tokenize(chunk_text)
                if not tokens:
                    continue
                chunk_id = f"{path.stem}-{chunk_index:03d}"
                token_counts = Counter(tokens)
                chunks.append(
                    _KnowledgeChunk(
                        chunk_id=chunk_id,
                        source_path=str(path),
                        title=section_title or doc_title,
                        text=chunk_text,
                        tokens=tokens,
                        token_counts=dict(token_counts),
                        length=max(len(tokens), 1),
                    )
                )
                chunk_index += 1
        return chunks

    def _extract_doc_title(self, path: Path, text: str) -> str:
        for line in text.splitlines():
            match = SECTION_RE.match(line)
            if match:
                return match.group(1).strip()
        return path.stem

    def _split_sections(self, text: str, default_title: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_title = default_title
        buffer: list[str] = []

        def flush():
            joined = self._normalize_text("\n".join(buffer)).strip()
            if joined:
                sections.append((current_title, joined))

        for line in text.splitlines():
            match = SECTION_RE.match(line)
            if match:
                flush()
                current_title = match.group(1).strip()
                buffer = []
                continue
            buffer.append(line)

        flush()
        if not sections and text.strip():
            sections.append((default_title, text.strip()))
        return sections

    def _chunk_text(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = f"{current}\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self.config.chunk_max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(paragraph) <= self.config.chunk_max_chars:
                current = paragraph
                continue

            sentences = [part.strip() for part in SENTENCE_RE.split(paragraph) if part.strip()]
            sentence_buffer = ""
            for sentence in sentences:
                merged = f"{sentence_buffer}{sentence}".strip()
                if len(merged) <= self.config.chunk_max_chars:
                    sentence_buffer = merged
                else:
                    if sentence_buffer:
                        chunks.append(sentence_buffer)
                    sentence_buffer = ""
                    remaining = sentence.strip()
                    while len(remaining) > self.config.chunk_max_chars:
                        chunks.append(remaining[: self.config.chunk_max_chars].strip())
                        remaining = remaining[self.config.chunk_max_chars :].strip()
                    sentence_buffer = remaining
            if sentence_buffer:
                chunks.append(sentence_buffer)

        if current:
            chunks.append(current)
        return chunks

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _tokenize(self, text: str) -> list[str]:
        terms: list[str] = []
        lowered = text.lower()
        raw_tokens = (
            jieba.cut_for_search(lowered)
            if jieba is not None
            else re.findall(r"[a-z0-9_./:-]+|[\u4e00-\u9fff]{1,4}", lowered)
        )
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            if token in STOPWORDS:
                continue
            if re.fullmatch(r"[a-z0-9_./:-]+", token):
                if len(token) >= 2:
                    terms.append(token)
                continue
            if re.search(r"[\u4e00-\u9fff]", token):
                if len(token) >= 1:
                    terms.append(token)
        return terms

    def _idf(self, term: str) -> float:
        doc_count = len(self._chunks)
        freq = self._doc_freq.get(term, 0)
        return log(1 + (doc_count - freq + 0.5) / (freq + 0.5)) if doc_count else 0.0

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self.config.enabled:
            return []

        query = query.strip()
        if not query:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        top_k = top_k or self.config.top_k
        scored: list[RetrievedChunk] = []
        unique_query_terms = set(query_tokens)
        lowered_query = query.lower()

        for chunk in self._chunks:
            overlap_terms = sorted(unique_query_terms.intersection(chunk.token_counts))
            if not overlap_terms:
                continue

            score = self._bm25_score(chunk, unique_query_terms)
            if lowered_query in chunk.text.lower():
                score += 2.5
            score += min(len(overlap_terms) * 0.18, 1.2)
            if score < self.config.min_score:
                continue

            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    title=chunk.title,
                    text=chunk.text,
                    score=round(score, 4),
                    overlap_terms=overlap_terms[:8],
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _bm25_score(self, chunk: _KnowledgeChunk, query_terms: set[str]) -> float:
        score = 0.0
        k1 = 1.5
        b = 0.75
        avg_len = self._avg_doc_len or 1.0
        for term in query_terms:
            freq = chunk.token_counts.get(term, 0)
            if not freq:
                continue
            idf = self._idf(term)
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * chunk.length / avg_len)
            score += idf * numerator / denominator
        return score

    def get_status(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "knowledge_dir": str(self.config.knowledge_dir),
            "knowledge_dir_exists": self.config.knowledge_dir.exists(),
            "index_path": str(self.config.index_path),
            "document_count": len(self._documents),
            "chunk_count": len(self._chunks),
            "top_k": self.config.top_k,
            "last_build_at": self._last_build_at,
            "loaded_from_cache": self._index_loaded_from_cache,
        }

    def _write_cache(self):
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "built_at": self._last_build_at,
            "knowledge_dir": str(self.config.knowledge_dir),
            "documents": self._document_snapshots(),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "title": chunk.title,
                    "text": chunk.text,
                    "tokens": chunk.tokens,
                    "token_counts": chunk.token_counts,
                    "length": chunk.length,
                }
                for chunk in self._chunks
            ],
        }
        self.config.index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _try_load_cache(self) -> bool:
        if not self.config.index_path.exists():
            return False

        try:
            payload = json.loads(self.config.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        if payload.get("documents") != self._document_snapshots():
            return False

        chunks: list[_KnowledgeChunk] = []
        for item in payload.get("chunks", []):
            chunks.append(
                _KnowledgeChunk(
                    chunk_id=item["chunk_id"],
                    source_path=item["source_path"],
                    title=item["title"],
                    text=item["text"],
                    tokens=item["tokens"],
                    token_counts=item["token_counts"],
                    length=int(item["length"]),
                )
            )

        self._chunks = chunks
        self._documents = [item["path"] for item in payload.get("documents", [])]
        self._doc_freq = Counter()
        total_length = 0
        for chunk in self._chunks:
            total_length += chunk.length
            for token in set(chunk.tokens):
                self._doc_freq[token] += 1

        self._avg_doc_len = total_length / len(self._chunks) if self._chunks else 0.0
        self._last_build_at = payload.get("built_at", "")
        self._index_loaded_from_cache = True
        return True

    def _document_snapshots(self) -> list[dict]:
        snapshots = []
        for path in sorted(self._iter_knowledge_files()):
            stat = path.stat()
            snapshots.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return snapshots
