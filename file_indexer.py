"""
file_indexer.py — Jarvis PC file indexing system
Walks a directory tree, reads text files, chunks them, and stores
embeddings + metadata in ChromaDB via MemoryEngine.

Features:
  - Configurable root path, max file size, included extensions
  - Text chunking (overlapping windows for better semantic recall)
  - Path + filename metadata stored with each chunk
  - Progress callback for UI updates
  - Graceful permission / encoding error handling
  - Skips hidden files and common noise directories (node_modules etc.)
  - Can be run as __main__ for standalone indexing

Usage (from code):
    from file_indexer import FileIndexer
    from memory import MemoryEngine

    mem     = MemoryEngine()
    indexer = FileIndexer(mem)
    count   = indexer.index_directory("C:/Users/Me/Documents")
    print(f"Indexed {count} chunks")

Usage (CLI):
    python file_indexer.py --path ~/Documents --extensions .txt .md .py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

from jarvis_safety import (
    SAFE_TEXT_EXTENSIONS,
    safe_iter_files,
    validate_read_path,
    validate_text_file,
)

# ── Default configuration ──────────────────────────────────────────────────────

DEFAULT_EXTENSIONS = set(SAFE_TEXT_EXTENSIONS)

MAX_FILE_SIZE_BYTES = 512_000   # 512 KB per file
CHUNK_SIZE          = 800       # characters per chunk
CHUNK_OVERLAP       = 120       # overlap to preserve context across chunks


class FileIndexer:
    """
    Indexes a directory tree into MemoryEngine (ChromaDB).

    Args:
        memory:         MemoryEngine instance (must have available=True)
        extensions:     Set of file extensions to process
        max_file_size:  Skip files larger than this (bytes)
        chunk_size:     Characters per text chunk
        chunk_overlap:  Overlap between consecutive chunks
    """

    def __init__(
        self,
        memory,
        extensions: Optional[set[str]] = None,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        chunk_size:    int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.memory        = memory
        self.extensions    = extensions or DEFAULT_EXTENSIONS
        self.max_file_size = max_file_size
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ─────────────────────────────────────────────────────────────

    def index_directory(
        self,
        root: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """
        Walk `root` and index all matching files.

        Args:
            root:        Path to directory
            progress_cb: Optional callback(files_done, files_total, current_path)

        Returns:
            Number of chunks stored.
        """
        if not self.memory.available:
            print("[FileIndexer] Memory engine unavailable — aborting.")
            return 0

        ok, msg, root_p = validate_read_path(root)
        if not ok:
            print(f"[FileIndexer] {msg}")
            return 0
        if not root_p.is_dir():
            print(f"[FileIndexer] Not a directory: {root_p}")
            return 0

        # Collect files first so we can report progress
        files = list(self._walk(root_p))
        total = len(files)
        if total == 0:
            print(f"[FileIndexer] No matching files found under {root_p}")
            return 0

        print(f"[FileIndexer] Found {total} file(s) to index under {root_p}")
        chunks_stored = 0

        for idx, fp in enumerate(files):
            if progress_cb:
                progress_cb(idx, total, str(fp))
            chunks_stored += self._index_file(fp)

        if progress_cb:
            progress_cb(total, total, "done")

        print(f"[FileIndexer] Done — {chunks_stored} chunk(s) stored from {total} file(s).")
        return chunks_stored

    def index_file_single(self, path: str) -> int:
        """Index a single file.  Returns number of chunks stored."""
        if not self.memory.available:
            return 0
        ok, _, fp = validate_read_path(path, must_be_file=True)
        if not ok:
            return 0
        return self._index_file(fp)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _walk(self, root: Path):
        """Yield file paths, skipping noise directories and unsupported types."""
        for entry in safe_iter_files(root):
            if entry.suffix.lower() not in self.extensions:
                continue
            ok, _ = validate_text_file(entry)
            if not ok:
                continue
            try:
                if entry.stat().st_size > self.max_file_size:
                    continue
            except OSError:
                continue
            yield entry

    def _index_file(self, fp: Path) -> int:
        """Read, chunk, and store one file.  Returns chunks stored."""
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"[FileIndexer] Cannot read {fp}: {exc}")
            return 0

        text = text.strip()
        if not text:
            return 0

        chunks = self._chunk(text)
        stored = 0
        for i, chunk in enumerate(chunks):
            meta = {
                "type":       "file",
                "path":       str(fp),
                "filename":   fp.name,
                "extension":  fp.suffix.lower(),
                "chunk_idx":  i,
                "chunk_total": len(chunks),
            }
            self.memory.store_file_chunk(str(fp), chunk, chunk_idx=i, extra_meta=meta)
            stored += 1

        return stored

    def _chunk(self, text: str) -> list[str]:
        """
        Split text into overlapping fixed-size chunks.
        Falls back to the full text if shorter than chunk_size.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start  = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks


# ── CLI entry point ────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="JARVIS File Indexer — embed your filesystem into ChromaDB"
    )
    parser.add_argument("--path",       default="~/Documents",
                        help="Root directory to index (default: ~/Documents)")
    parser.add_argument("--db",         default="./db",
                        help="ChromaDB path (default: ./db)")
    parser.add_argument("--extensions", nargs="*",
                        help="File extensions to include (e.g. .txt .md .py)")
    args = parser.parse_args()

    # Import here so the module can be imported without ChromaDB being available
    from memory import MemoryEngine
    mem = MemoryEngine(db_path=args.db)
    if not mem.available:
        print("ChromaDB / sentence-transformers not installed.")
        print("pip install chromadb sentence-transformers")
        sys.exit(1)

    exts = set(args.extensions) if args.extensions else None
    indexer = FileIndexer(mem, extensions=exts)

    def progress(done, total, path):
        pct = int(done / total * 100) if total else 100
        print(f"\r  [{pct:3d}%] ({done}/{total})  {Path(path).name[:50]:<50}", end="", flush=True)
        if done == total:
            print()

    count = indexer.index_directory(args.path, progress_cb=progress)
    print(f"\nTotal chunks indexed: {count}")


if __name__ == "__main__":
    _cli()
