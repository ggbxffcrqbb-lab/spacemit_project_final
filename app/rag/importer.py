from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from xml.etree import ElementTree
import zipfile

from app.core.config import RagConfig
from app.rag.knowledge_base import LocalKnowledgeBase


ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")


@dataclass
class ImportedDocument:
    source_path: str
    output_path: str
    title: str
    category: str
    tags: list[str]
    content_hash: str
    bytes_written: int


class KnowledgeImporter:
    def __init__(self, config: RagConfig):
        self.config = config

    def import_sources(
        self,
        sources: list[str],
        recursive: bool = False,
        category: str = "",
        tags: list[str] | None = None,
        title_prefix: str = "",
        dest_subdir: str = "imported",
        rebuild: bool = True,
        replace: bool = False,
    ) -> dict:
        tags = [tag.strip() for tag in (tags or []) if tag.strip()]
        dest_dir = self._resolve_dest_dir(dest_subdir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        expanded_sources, skipped = self._expand_sources(sources, recursive=recursive)
        imported: list[ImportedDocument] = []
        seen_output_paths: set[str] = set()
        seen_hashes: set[str] = set()

        for source in expanded_sources:
            try:
                text = self._read_source_text(source)
                normalized = self._normalize_text(text)
                if not normalized:
                    skipped.append({"source_path": str(source), "reason": "empty_after_normalize"})
                    continue

                content_hash = sha1(normalized.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    skipped.append({"source_path": str(source), "reason": "duplicate_content"})
                    continue
                seen_hashes.add(content_hash)

                title = self._derive_title(source, normalized, category, title_prefix)
                output_path = self._build_output_path(
                    source=source,
                    dest_dir=dest_dir,
                    title=title,
                    normalized=normalized,
                    replace=replace,
                )
                if str(output_path) in seen_output_paths:
                    skipped.append({"source_path": str(source), "reason": "duplicate_output_path"})
                    continue
                seen_output_paths.add(str(output_path))

                body = self._build_output_body(
                    title=title,
                    normalized=normalized,
                    source=source,
                    category=category,
                    tags=tags,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(body, encoding="utf-8")

                imported.append(
                    ImportedDocument(
                        source_path=str(source),
                        output_path=str(output_path),
                        title=title,
                        category=category,
                        tags=tags,
                        content_hash=content_hash,
                        bytes_written=len(body.encode("utf-8")),
                    )
                )
            except Exception as exc:
                skipped.append({"source_path": str(source), "reason": str(exc)})

        kb = LocalKnowledgeBase(self.config)
        if rebuild:
            kb.rebuild()
        catalog = self._write_catalog(imported, skipped)

        return {
            "imported_count": len(imported),
            "skipped_count": len(skipped),
            "dest_dir": str(dest_dir),
            "rebuild_triggered": rebuild,
            "catalog_path": str(catalog),
            "documents": [doc.__dict__ for doc in imported],
            "skipped": skipped,
            "rag_status": kb.get_status(),
        }

    def _resolve_dest_dir(self, dest_subdir: str) -> Path:
        subdir = Path(dest_subdir)
        if subdir.is_absolute():
            raise ValueError("dest_subdir must be relative to knowledge_dir")

        knowledge_root = self.config.knowledge_dir.resolve()
        dest_dir = (knowledge_root / subdir).resolve()
        if dest_dir != knowledge_root and knowledge_root not in dest_dir.parents:
            raise ValueError("dest_subdir escapes knowledge_dir")
        return dest_dir

    def _expand_sources(self, sources: list[str], recursive: bool) -> tuple[list[Path], list[dict]]:
        expanded: list[Path] = []
        skipped: list[dict] = []

        for raw in sources:
            path = Path(raw).expanduser()
            if not path.exists():
                skipped.append({"source_path": str(path), "reason": "not_found"})
                continue

            if path.is_file():
                if path.suffix.lower() in ALLOWED_EXTENSIONS:
                    expanded.append(path.resolve())
                else:
                    skipped.append({"source_path": str(path), "reason": "unsupported_extension"})
                continue

            iterator = path.rglob("*") if recursive else path.glob("*")
            matched = 0
            for child in iterator:
                if child.is_file() and child.suffix.lower() in ALLOWED_EXTENSIONS:
                    expanded.append(child.resolve())
                    matched += 1
            if matched == 0:
                skipped.append({"source_path": str(path), "reason": "no_supported_files"})

        unique = sorted(set(expanded))
        return list(unique), skipped

    def _read_source_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            return self._read_text_file(path)
        if suffix == ".pdf":
            return self._read_pdf_text(path)
        if suffix == ".docx":
            return self._read_docx_text(path)
        raise ValueError(f"unsupported extension: {suffix}")

    def _read_text_file(self, path: Path) -> str:
        last_error: Exception | None = None
        for encoding in TEXT_ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError(f"unable to decode text file: {path}") from last_error
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf_text(self, path: Path) -> str:
        pdftotext_path = shutil.which("pdftotext")
        if pdftotext_path:
            proc = subprocess.run(
                [pdftotext_path, str(path), "-"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            if proc.returncode != 0:
                raise RuntimeError(f"pdftotext failed: {proc.stderr.strip()}")
            return proc.stdout

        try:
            from pypdf import PdfReader  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("pdf import requires pdftotext or pypdf") from exc

        reader = PdfReader(str(path))
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)

    def _read_docx_text(self, path: Path) -> str:
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        try:
            with zipfile.ZipFile(path) as archive:
                document_xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise RuntimeError("docx file missing word/document.xml") from exc
        except zipfile.BadZipFile as exc:
            raise RuntimeError("invalid docx file") from exc

        root = ElementTree.fromstring(document_xml)
        paragraphs: list[str] = []
        for node in root.findall(".//w:p", namespace):
            parts = [text_node.text or "" for text_node in node.findall(".//w:t", namespace)]
            line = "".join(parts).strip()
            if line:
                paragraphs.append(line)
        return "\n".join(paragraphs)

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.splitlines()]
        compact_lines = []
        previous_blank = False
        for line in lines:
            is_blank = not line
            if is_blank and previous_blank:
                continue
            compact_lines.append(line)
            previous_blank = is_blank
        return "\n".join(compact_lines).strip()

    def _derive_title(self, source: Path, normalized: str, category: str, title_prefix: str) -> str:
        first_line = ""
        for line in normalized.splitlines():
            if not line:
                continue
            first_line = line.lstrip("#").strip()
            break

        title = first_line if 2 <= len(first_line) <= 48 else source.stem
        if category:
            title = f"{category} - {title}"
        if title_prefix:
            title = f"{title_prefix}{title}"
        return title.strip()

    def _build_output_path(
        self,
        source: Path,
        dest_dir: Path,
        title: str,
        normalized: str,
        replace: bool,
    ) -> Path:
        slug = self._slugify(title) or self._slugify(source.stem) or "knowledge"
        digest = sha1(normalized.encode("utf-8")).hexdigest()[:8]
        filename = f"{slug}.md" if replace else f"{slug}-{digest}.md"
        return dest_dir / filename

    def _build_output_body(
        self,
        title: str,
        normalized: str,
        source: Path,
        category: str,
        tags: list[str],
    ) -> str:
        metadata = [
            f"来源文件：{source}",
            f"导入时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if category:
            metadata.append(f"分类：{category}")
        if tags:
            metadata.append(f"标签：{', '.join(tags)}")

        return (
            f"# {title}\n\n"
            "## 导入元数据\n\n"
            + "\n".join(f"- {line}" for line in metadata)
            + "\n\n## 正文\n\n"
            + normalized
            + "\n"
        )

    def _write_catalog(self, imported: list[ImportedDocument], skipped: list[dict]) -> Path:
        catalog_path = self.config.index_path.parent / "knowledge_import_catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "knowledge_dir": str(self.config.knowledge_dir),
            "imported": [doc.__dict__ for doc in imported],
            "skipped": skipped,
        }
        catalog_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return catalog_path

    def _slugify(self, value: str) -> str:
        value = value.strip().lower()
        value = value.replace(" ", "-")
        value = re.sub(r"[^0-9a-z\u4e00-\u9fff_-]+", "-", value)
        value = re.sub(r"-{2,}", "-", value).strip("-_")
        return value[:80]
