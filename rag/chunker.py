"""Recursive text chunker for document splitting.

Strategy:
1. Split by markdown headers (##, ###) — preserve section boundaries
2. Split by paragraphs (double newline)
3. Split by sentences if paragraph is still too large
4. Enforce chunk_size limit with overlap

Default: ~500 tokens per chunk with 50-token overlap (good for embedding models).
"""

import re
from typing import Optional
from loguru import logger


class TextChunker:
    """Recursive text splitter tuned for markdown reports and search results."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---- public API ----

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[dict]:
        """Split text into overlapping chunks with metadata.

        Returns list of dicts with keys: content, index, metadata
        """
        if not text or not text.strip():
            return []

        meta = dict(metadata or {})

        # Stage 1: split by markdown sections
        sections = self._split_by_sections(text)

        # Stage 2: split each section to fit chunk_size
        chunks: list[dict] = []
        for section_idx, section in enumerate(sections):
            section_chunks = self._split_section(section)
            for ch in section_chunks:
                chunks.append({
                    "content": ch,
                    "chunk_index": len(chunks),
                    "section_index": section_idx,
                    "metadata": meta,
                })

        # Add overlap: each chunk includes tail of previous chunk
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        logger.debug(f"Chunked text ({len(text)} chars) → {len(chunks)} chunks")
        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """Chunk multiple documents. Each doc: {content: str, metadata: dict}"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(doc["content"], doc.get("metadata"))
            for ch in chunks:
                ch["doc_index"] = len(all_chunks)
                all_chunks.append(ch)
        return all_chunks

    # ---- internal ----

    def _split_by_sections(self, text: str) -> list[str]:
        """Split by markdown headers while preserving header in each section."""
        # Match ## or ### headers
        pattern = r'(?=^#{2,3}\s)'
        parts = re.split(pattern, text, flags=re.MULTILINE)

        # If no headers found, treat as single section
        if len(parts) <= 1:
            return [text.strip()]

        # Combine consecutive header-only lines with their content
        sections: list[str] = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # If this is a header line followed by content, OR it's a header fragment
            if re.match(r'^#{2,3}\s', part):
                if current:
                    sections.append(current)
                current = part
            else:
                if current:
                    current += "\n\n" + part
                else:
                    current = part

        if current:
            sections.append(current)

        return sections or [text.strip()]

    def _split_section(self, text: str) -> list[str]:
        """Split a single section into chunks within size limit."""
        # Rough token estimate: 1 token ≈ 4 chars for Chinese, 4 chars ≈ 1 token for English
        max_chars = self.chunk_size * 4

        if len(text) <= max_chars:
            return [text]

        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current and len(current) + len(para) + 2 > max_chars:
                if current.strip():
                    chunks.append(current.strip())
                current = para
            else:
                current = (current + "\n\n" + para) if current else para

            # If a single paragraph is too long, split by sentences
            while len(current) > max_chars:
                split_point = self._find_split_point(current, max_chars)
                chunks.append(current[:split_point].strip())
                current = current[split_point:].strip()

        if current.strip():
            chunks.append(current.strip())

        return chunks

    @staticmethod
    def _find_split_point(text: str, max_chars: int) -> int:
        """Find a natural split point near max_chars."""
        # Try to split at sentence boundary (。！？\n)
        search_start = max(0, max_chars - 100)
        for pattern in ['。', '！', '？', '\n', '；', '. ']:
            idx = text.rfind(pattern, search_start, max_chars + 50)
            if idx != -1 and idx > search_start:
                return idx + 1

        # Fallback: split at word boundary (space)
        idx = text.rfind(' ', search_start, max_chars + 50)
        if idx != -1 and idx > search_start:
            return idx + 1

        # Last resort: hard split
        return max_chars

    def _add_overlap(self, chunks: list[dict]) -> list[dict]:
        """Add overlap from previous chunk to each subsequent chunk."""
        overlap_chars = self.chunk_overlap * 4  # rough char estimate

        for i in range(1, len(chunks)):
            prev_content = chunks[i - 1]["content"]
            if len(prev_content) > overlap_chars:
                overlap_text = prev_content[-overlap_chars:]
                # Find the first sentence boundary in the overlap
                for pattern in ['。', '！', '？', '\n', '；', '. ']:
                    idx = overlap_text.find(pattern)
                    if idx != -1 and idx > 20:
                        overlap_text = overlap_text[idx + 1:]
                        break

                chunks[i]["content"] = overlap_text + "\n" + chunks[i]["content"]
                chunks[i]["overlap_from"] = i - 1

        return chunks
