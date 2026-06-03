"""
memory.py — Jarvis 3-layer cognitive memory engine
Stores and retrieves information across three distinct memory lanes:

  Layer 1 — Episodic:   conversation exchanges (what was said)
  Layer 2 — Semantic:   user facts / preferences (who the user is)
  Layer 3 — File:       indexed PC file content (what's on disk)

Each layer is a separate ChromaDB collection.
Falls back to a no-op stub if optional dependencies are missing.

FIXES APPLIED:
  - [FIX-6] retrieve(): cosine distance threshold (< 0.35) filters irrelevant results
  - [UPGRADE] 3-layer architecture with per-layer retrieval and injection
  - [UPGRADE] Semantic memory: structured key/value user facts
  - [UPGRADE] inject_context: structured prompt with labelled sections
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Optional

from jarvis_safety import validate_read_path, validate_text_file

# ── Optional heavy imports ─────────────────────────────────────────────────────
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# Cosine distance threshold.  Range: 0 = identical, 2 = opposite.
_RELEVANCE_THRESHOLD = 0.35

# Collection names
_COL_EPISODIC = "jarvis_episodic"
_COL_SEMANTIC = "jarvis_semantic"
_COL_FILE     = "jarvis_files"


class MemoryEngine:
    """
    3-layer semantic memory backed by ChromaDB + sentence-transformers.

    Layer 1 — Episodic  : conversation history
    Layer 2 — Semantic  : user facts / preferences
    Layer 3 — File      : indexed PC files

    All public methods degrade silently when dependencies are missing.
    """

    def __init__(self, db_path: str = "./db"):
        self.available    = _DEPS_OK
        self._embedder    = None
        self._client      = None
        self._col_ep      = None   # episodic
        self._col_sem     = None   # semantic / user profile
        self._col_file    = None   # file index

        if not self.available:
            print("[Memory] ChromaDB / sentence-transformers not found — memory disabled.")
            print("[Memory] Install:  pip install chromadb sentence-transformers")
            return

        os.makedirs(db_path, exist_ok=True)

        print("[Memory] Connecting to ChromaDB …")
        self._client = chromadb.PersistentClient(path=db_path)

        hnsw = {"hnsw:space": "cosine"}
        self._col_ep   = self._client.get_or_create_collection(_COL_EPISODIC, metadata=hnsw)
        self._col_sem  = self._client.get_or_create_collection(_COL_SEMANTIC, metadata=hnsw)
        self._col_file = self._client.get_or_create_collection(_COL_FILE,     metadata=hnsw)

        print("[Memory] Loading embedding model (all-MiniLM-L6-v2) …")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        print(
            f"[Memory] Ready. "
            f"Episodic={self._col_ep.count()}  "
            f"Semantic={self._col_sem.count()}  "
            f"Files={self._col_file.count()}"
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        return self._embedder.encode(text).tolist()

    def _make_id(self, text: str) -> str:
        return hashlib.md5(
            f"{text}{datetime.now().isoformat()}".encode()
        ).hexdigest()

    def _query_col(self, col, query: str, n: int) -> list[str]:
        """Query a collection and return docs passing the relevance threshold."""
        if col is None or col.count() == 0:
            return []
        n = min(n, col.count())
        results = col.query(
            query_embeddings=[self._embed(query)],
            n_results=n,
            include=["documents", "distances"],
        )
        docs      = results.get("documents", [[]])[0]
        distances = results.get("distances",  [[]])[0]
        return [d for d, dist in zip(docs, distances) if dist < _RELEVANCE_THRESHOLD]

    # ── Layer 1: Episodic memory ───────────────────────────────────────────────

    def store(
        self,
        user_msg: str,
        assistant_msg: str,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """Store one conversation exchange in episodic memory."""
        if not self.available:
            return
        text = f"User: {user_msg}\nJarvis: {assistant_msg}"
        meta = {
            "timestamp":  datetime.now().isoformat(),
            "user_msg":   user_msg[:300],
            "layer":      "episodic",
            **(extra_meta or {}),
        }
        self._col_ep.add(
            documents=[text],
            embeddings=[self._embed(text)],
            metadatas=[meta],
            ids=[self._make_id(text)],
        )

    def retrieve(self, query: str, n_results: int = 3) -> list[str]:
        """Retrieve relevant episodic memories."""
        if not self.available:
            return []
        return self._query_col(self._col_ep, query, n_results)

    # ── Layer 2: Semantic / user profile memory ────────────────────────────────

    def store_fact(self, key: str, value: str, category: str = "preference") -> None:
        """
        Store a structured user fact in semantic memory.

        Args:
            key:      Short label (e.g. "music genre", "occupation")
            value:    The fact value (e.g. "country", "software engineer")
            category: One of: preference, habit, goal, identity, other
        """
        if not self.available:
            return
        text = f"{key}: {value}"
        meta = {
            "timestamp": datetime.now().isoformat(),
            "key":       key,
            "value":     value[:500],
            "category":  category,
            "layer":     "semantic",
        }
        # Upsert by key — overwrite existing fact with same key
        doc_id = hashlib.md5(key.lower().encode()).hexdigest()
        existing = self._col_sem.get(ids=[doc_id])
        if existing and existing.get("ids"):
            self._col_sem.update(
                ids=[doc_id],
                documents=[text],
                embeddings=[self._embed(text)],
                metadatas=[meta],
            )
        else:
            self._col_sem.add(
                documents=[text],
                embeddings=[self._embed(text)],
                metadatas=[meta],
                ids=[doc_id],
            )

    def retrieve_facts(self, query: str, n_results: int = 5) -> list[str]:
        """Retrieve relevant user facts from semantic memory."""
        if not self.available:
            return []
        return self._query_col(self._col_sem, query, n_results)

    def get_all_facts(self) -> list[dict]:
        """Return all stored user facts as a list of metadata dicts."""
        if not self.available or self._col_sem is None or self._col_sem.count() == 0:
            return []
        results = self._col_sem.get(include=["documents", "metadatas"])
        return [
            {"text": doc, **meta}
            for doc, meta in zip(
                results.get("documents", []),
                results.get("metadatas", []),
            )
        ]

    def auto_extract_facts(self, user_msg: str, assistant_msg: str) -> None:
        """
        Heuristically extract user facts from a conversation turn and
        store them in semantic memory.  Runs silently — never raises.
        """
        if not self.available:
            return
        import re
        text = user_msg.lower()

        patterns = [
            # "I am / I'm a ..."
            (r"\bi(?:'m| am) (?:a |an )?([a-z ]{3,40})", "occupation", "identity"),
            # "I work as ..."
            (r"\bi work (?:as |in )?([a-z ]{3,40})", "occupation", "identity"),
            # "I live in ..."
            (r"\bi live in ([a-z ,]{3,40})", "location", "identity"),
            # "I like / love / prefer ..."
            (r"\bi (?:like|love|enjoy|prefer) ([a-z ]{3,40})", "preference", "preference"),
            # "my name is ..."
            (r"\bmy name is ([a-z]{2,30})", "name", "identity"),
            # "I usually / always ..."
            (r"\bi (?:usually|always|often) ([a-z ]{3,40})", "habit", "habit"),
        ]

        for pattern, key, category in patterns:
            m = re.search(pattern, text)
            if m:
                value = m.group(1).strip().rstrip(".,!?")
                if len(value) > 2:
                    try:
                        self.store_fact(key, value, category)
                    except Exception:
                        pass

    # ── Layer 3: File memory ───────────────────────────────────────────────────

    def store_file_chunk(
        self,
        file_path: str,
        chunk_text: str,
        chunk_idx: int = 0,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """Store one chunk from an indexed file."""
        if not self.available:
            return
        doc_id = hashlib.md5(
            f"{file_path}:{chunk_idx}:{chunk_text[:80]}".encode()
        ).hexdigest()
        meta = {
            "timestamp":  datetime.now().isoformat(),
            "path":       file_path,
            "chunk_idx":  chunk_idx,
            "layer":      "file",
            **(extra_meta or {}),
        }
        self._col_file.upsert(
            documents=[chunk_text],
            embeddings=[self._embed(chunk_text)],
            metadatas=[meta],
            ids=[doc_id],
        )

    def retrieve_files(self, query: str, n_results: int = 3) -> list[str]:
        """Retrieve relevant file chunks."""
        if not self.available:
            return []
        return self._query_col(self._col_file, query, n_results)

    def index_file(self, path: str) -> bool:
        """
        Index a single text file (simple fallback — FileIndexer is preferred
        for proper chunking and progress reporting).
        """
        if not self.available:
            return False
        try:
            ok, msg, p = validate_read_path(path, must_be_file=True)
            if not ok:
                print(f"[Memory] {msg}")
                return False
            ok, msg = validate_text_file(p)
            if not ok:
                print(f"[Memory] {msg}")
                return False
            with open(p, "r", errors="replace") as fh:
                content = fh.read(8000)
            self.store_file_chunk(str(p), content, chunk_idx=0,
                                  extra_meta={"filename": os.path.basename(p)})
            return True
        except Exception as exc:
            print(f"[Memory] Could not index {path}: {exc}")
            return False

    # ── Context injection (3-layer prompt builder) ─────────────────────────────

    def inject_context(self, query: str, system_prompt: str) -> str:
        """
        Build an enriched system prompt with three labelled memory sections:
            [USER PROFILE]          — semantic facts about the user
            [RELEVANT MEMORIES]     — episodic conversation history
            [RELEVANT FILE CONTEXT] — file index hits

        Returns the original prompt if all layers are empty for this query.
        """
        if not self.available:
            return system_prompt

        sections: list[str] = []

        # Layer 2 — always include relevant user facts
        facts = self.retrieve_facts(query, n_results=5)
        if facts:
            block = "\n".join(f"  • {f[:300]}" for f in facts)
            sections.append(f"[USER PROFILE]\n{block}")

        # Layer 1 — episodic conversation history
        memories = self.retrieve(query, n_results=3)
        if memories:
            block = "\n".join(f"  • {m[:400]}" for m in memories)
            sections.append(f"[RELEVANT MEMORIES]\n{block}")

        # Layer 3 — file index
        file_hits = self.retrieve_files(query, n_results=3)
        if file_hits:
            block = "\n".join(f"  • {h[:400]}" for h in file_hits)
            sections.append(f"[RELEVANT FILE CONTEXT]\n{block}")

        if not sections:
            return system_prompt

        return system_prompt + "\n\n" + "\n\n".join(sections)

    # ── Housekeeping ───────────────────────────────────────────────────────────

    def clear(self, layer: str = "episodic") -> None:
        """
        Clear a specific memory layer (episodic / semantic / file / all).
        """
        if not self.available:
            return

        hnsw = {"hnsw:space": "cosine"}
        if layer in ("episodic", "all"):
            try:
                self._client.delete_collection(_COL_EPISODIC)
            except Exception:
                pass
            self._col_ep = self._client.get_or_create_collection(_COL_EPISODIC, metadata=hnsw)
            print("[Memory] Episodic memory cleared.")
        if layer in ("semantic", "all"):
            try:
                self._client.delete_collection(_COL_SEMANTIC)
            except Exception:
                pass
            self._col_sem = self._client.get_or_create_collection(_COL_SEMANTIC, metadata=hnsw)
            print("[Memory] Semantic memory cleared.")
        if layer in ("file", "all"):
            try:
                self._client.delete_collection(_COL_FILE)
            except Exception:
                pass
            self._col_file = self._client.get_or_create_collection(_COL_FILE, metadata=hnsw)
            print("[Memory] File memory cleared.")

    def count(self, layer: str = "episodic") -> int:
        """Count entries in a layer (episodic / semantic / file / all)."""
        if not self.available:
            return 0
        if layer == "episodic":
            return self._col_ep.count()  if self._col_ep  else 0
        if layer == "semantic":
            return self._col_sem.count() if self._col_sem else 0
        if layer == "file":
            return self._col_file.count() if self._col_file else 0
        if layer == "all":
            ep  = self._col_ep.count()   if self._col_ep   else 0
            sem = self._col_sem.count()  if self._col_sem  else 0
            fil = self._col_file.count() if self._col_file else 0
            return ep + sem + fil
        return 0
