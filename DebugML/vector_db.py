"""
Production-ready vector database module using FAISS for ML failure retrieval.

Persistence: FAISS index and metadata are saved to disk. On startup, the index
is loaded automatically if it exists and source data is unchanged—avoiding
embedding recomputation on every run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

import faiss
import numpy as np

from embeddings import EmbeddingModel, EmbeddingError

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "failures.json"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "data" / "vector_db"
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class FailureRecord(TypedDict):
    """Type definition for a failure record from the dataset."""

    id: int
    failure_description: str
    root_cause: str
    solution: str
    tags: list[str]


class SearchResult(TypedDict):
    """Type definition for a search result with similarity score."""

    failure: FailureRecord
    score: float
    rank: int


class VectorDBError(Exception):
    """Raised when vector database operations fail."""

    pass


def _failure_to_text(failure: FailureRecord) -> str:
    """Combine failure fields into a single searchable text."""
    parts = [
        failure.get("failure_description", ""),
        failure.get("root_cause", ""),
        failure.get("solution", ""),
        " ".join(failure.get("tags", [])),
    ]
    return " | ".join(p for p in parts if p)


class VectorDB:
    """
    FAISS-backed vector database for ML failure similarity search.

    Persists index to disk (data/vector_db/index.faiss + metadata.json).
    Loads automatically on init when index exists and source data is unchanged.
    Embeddings are recomputed only when building a new index or source data changed.
    """

    def __init__(
        self,
        data_path: Path | str | None = None,
        index_dir: Path | str | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """
        Initialize the vector database.

        Loads persisted index from disk if it exists and data is unchanged.
        Otherwise builds from JSON, computes embeddings, and saves to disk.

        Args:
            data_path: Path to failures.json. Default: data/failures.json
            index_dir: Directory for persisted index.faiss and metadata.json
            force_rebuild: If True, rebuild index even if valid index exists.
        """
        self._data_path = Path(data_path or DEFAULT_DATA_PATH)
        self._index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        self._index_path = self._index_dir / INDEX_FILENAME
        self._metadata_path = self._index_dir / METADATA_FILENAME

        self._embedder = EmbeddingModel()
        self._index: faiss.IndexFlatIP | None = None
        self._failures: list[FailureRecord] = []

        self._load_or_build(force_rebuild=force_rebuild)

    def _is_index_stale(self) -> bool:
        """Return True if source data is newer than persisted index (needs rebuild)."""
        if not self._index_path.exists() or not self._data_path.exists():
            return True
        data_mtime = self._data_path.stat().st_mtime
        index_mtime = self._index_path.stat().st_mtime
        return data_mtime > index_mtime

    def _load_or_build(self, force_rebuild: bool = False) -> None:
        """
        Load existing index from disk if valid, otherwise build and persist.
        Avoids recomputing embeddings when index exists and source data unchanged.
        """
        if force_rebuild:
            logger.info("Force rebuild requested")
            self._build_index()
            return

        if self._index_path.exists() and self._metadata_path.exists() and not self._is_index_stale():
            try:
                self._load_index()
                logger.info(
                    "Loaded FAISS index from %s (%d vectors, no embedding recomputation)",
                    self._index_dir,
                    self._index.ntotal,
                )
                return
            except Exception as e:
                logger.warning("Failed to load index, rebuilding: %s", e)

        self._build_index()

    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        self._index = faiss.read_index(str(self._index_path))

        with open(self._metadata_path, encoding="utf-8") as f:
            self._failures = json.load(f)

        if self._index.ntotal != len(self._failures):
            raise VectorDBError(
                f"Index size ({self._index.ntotal}) does not match metadata size ({len(self._failures)})"
            )

    def _build_index(self) -> None:
        """Load JSON, compute embeddings, create FAISS index, and persist to disk."""
        failures = self._load_dataset()
        if not failures:
            raise VectorDBError(f"No failures found in {self._data_path}")

        texts = [_failure_to_text(f) for f in failures]
        logger.info("Embedding %d failure records...", len(texts))

        embeddings = self._embedder.embed_batch(
            texts,
            batch_size=32,
            normalize=True,
            show_progress=True,
        )

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

        self._failures = failures

        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))

        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._failures, f, indent=2, ensure_ascii=False)

        logger.info(
            "Built and saved FAISS index to %s (%d vectors)",
            self._index_dir,
            self._index.ntotal,
        )

    def _load_dataset(self) -> list[FailureRecord]:
        """Load and validate failure dataset from JSON."""
        if not self._data_path.exists():
            raise VectorDBError(f"Dataset not found: {self._data_path}")

        try:
            with open(self._data_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.exception("Invalid JSON in %s", self._data_path)
            raise VectorDBError(f"Invalid JSON: {e}") from e

        if not isinstance(data, list):
            raise VectorDBError("Dataset must be a JSON array of failure objects")

        return [dict(f) for f in data]

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        """
        Find the most similar failures for a given query.

        Args:
            query: Natural language description of the ML problem or symptom.
            k: Number of top results to return. Default 3.

        Returns:
            List of SearchResult dicts with failure, score, and rank.
            Score is cosine similarity (higher = more similar).

        Raises:
            VectorDBError: If search fails.
        """
        if not query or not isinstance(query, str):
            raise VectorDBError("search() requires a non-empty string query")

        if self._index is None:
            raise VectorDBError("Index not loaded")

        k = min(k, self._index.ntotal)
        if k <= 0:
            return []

        try:
            query_embedding = self._embedder.embed(query, normalize=True)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

            scores, indices = self._index.search(query_embedding, k)

            results: list[SearchResult] = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
                if idx < 0:
                    continue
                results.append(
                    {
                        "failure": self._failures[idx],
                        "score": float(score),
                        "rank": rank,
                    }
                )

            return results

        except EmbeddingError:
            raise
        except Exception as e:
            logger.exception("Search failed for query (length=%d)", len(query))
            raise VectorDBError(f"Search failed: {e}") from e

    @property
    def size(self) -> int:
        """Number of failure records in the index."""
        return self._index.ntotal if self._index else 0
