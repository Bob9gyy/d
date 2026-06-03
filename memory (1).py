"""
memory.py — Jarvis persistent memory engine
Stores conversations as embeddings in ChromaDB.
Falls back to a no-op stub if the optional dependencies are missing.

FIXES APPLIED:
  - [FIX-6] retrieve(): filter results by cosine distance threshold (< 0.35)
             so irrelevant memories are never injected into the system prompt.
"""
import hashlib
import os
from datetime import datetime
from typing import Optional

# ── Optional heavy imports ─────────────────────────────────────────────────────
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# Cosine distance threshold for "relevant".
# Range: 0 = identical, 2 = opposite.  < 0.35 is a solid "related" cutoff.
_RELEVANCE_THRESHOLD = 0.35


class MemoryEngine:
    """
    Semantic memory backed by ChromaDB + sentence-transformers.

    If dependencies are not installed the engine silently degrades:
    all public methods return empty/zero values so the rest of the
    app works without changes.
    """

    def __init__(self, db_path: str = "./db"):
        self.available = _DEPS_OK
        self._embedder  = None
        self._client    = None
        self._col       = None

        if not self.available:
            print("[Memory] ChromaDB / sentence-transformers not found — memory disabled.")
            print("[Memory] Install:  pip install chromadb sentence-transformers")
            return

        os.makedirs(db_path, exist_ok=True)

        print("[Memory] Connecting to ChromaDB …")
        self._client = chromadb.PersistentClient(path=db_path)
        self._col = self._client.get_or_create_collection(
            name="jarvis_memory",
            metadata={"hnsw:space": "cosine"},
        )

        print("[Memory] Loading embedding model (all-MiniLM-L6-v2) …")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"[Memory] Ready. {self._col.count()} stored memories.")

    # ── Core operations ────────────────────────────────────────────────────────

    def store(
        self,
        user_msg: str,
        assistant_msg: str,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """Embed and persist one conversation exchange."""
        if not self.available:
            return

        text = f"User: {user_msg}\nJarvis: {assistant_msg}"
        doc_id = hashlib.md5(
            f"{text}{datetime.now().isoformat()}".encode()
        ).hexdigest()

        meta = {
            "timestamp":  datetime.now().isoformat(),
            "user_msg":   user_msg[:300],
            "type":       "conversation",
            **(extra_meta or {}),
        }

        embedding = self._embedder.encode(text).tolist()
        self._col.add(documents=[text], embeddings=[embedding],
                      metadatas=[meta], ids=[doc_id])

    def retrieve(self, query: str, n_results: int = 3) -> list[str]:
        """
        Return the N most semantically similar past exchanges.

        FIX-6: ChromaDB always returns exactly n_results matches even when
        none are actually related to the query.  We now request distances and
        filter out anything above _RELEVANCE_THRESHOLD (cosine distance ≥ 0.35),
        so the system prompt is never polluted with irrelevant old conversations.
        """
        if not self.available or self._col.count() == 0:
            return []

        embedding = self._embedder.encode(query).tolist()
        n = min(n_results, self._col.count())

        results = self._col.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "distances"],
        )

        docs      = results.get("documents", [[]])[0]
        distances = results.get("distances",  [[]])[0]

        # Keep only genuinely relevant results
        relevant = [
            doc for doc, dist in zip(docs, distances)
            if dist < _RELEVANCE_THRESHOLD
        ]
        return relevant

    def inject_context(self, query: str, system_prompt: str) -> str:
        """
        Retrieve relevant memories and prepend them to the system prompt.
        Returns the original prompt unchanged if no relevant memories exist.
        """
        memories = self.retrieve(query, n_results=3)
        if not memories:
            return system_prompt

        block = "\n".join(f"  • {m[:400]}" for m in memories)
        return (
            f"{system_prompt}\n\n"
            f"[RELEVANT PAST CONVERSATIONS]\n{block}\n"
        )

    def clear(self) -> None:
        """Delete all stored memories."""
        if not self.available:
            return
        self._client.delete_collection("jarvis_memory")
        self._col = self._client.get_or_create_collection(
            name="jarvis_memory",
            metadata={"hnsw:space": "cosine"},
        )
        print("[Memory] All memories cleared.")

    def count(self) -> int:
        """Number of stored memory entries."""
        if not self.available or self._col is None:
            return 0
        return self._col.count()

    # ── File-level indexing (Phase 3 stub) ────────────────────────────────────

    def index_file(self, path: str) -> bool:
        """
        Index a single text file into memory.
        Returns True on success.  (Full implementation in file_indexer.py.)
        """
        if not self.available:
            return False
        try:
            with open(path, "r", errors="replace") as fh:
                content = fh.read(8000)
            self.store(f"[FILE] {path}", content, extra_meta={"type": "file", "path": path})
            return True
        except Exception as exc:
            print(f"[Memory] Could not index {path}: {exc}")
            return False
