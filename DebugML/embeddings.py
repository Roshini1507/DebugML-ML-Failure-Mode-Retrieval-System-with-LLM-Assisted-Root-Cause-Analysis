"""Production-ready embedding module using sentence-transformers."""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when embedding operations fail."""

    pass


class EmbeddingModel:
    """
    Singleton wrapper for sentence-transformers embedding model.
    Uses all-MiniLM-L6-v2 for efficient, high-quality text embeddings.
    """

    _instance: EmbeddingModel | None = None
    _initialized: bool = False

    def __new__(cls, model_name: str = DEFAULT_MODEL, **kwargs) -> EmbeddingModel:
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the embedding model. Only loads once (singleton).

        Args:
            model_name: HuggingFace model ID. Default: all-MiniLM-L6-v2
            device: Device to run on ('cuda', 'cpu', or None for auto).
            **kwargs: Additional arguments passed to SentenceTransformer.
        """
        if EmbeddingModel._initialized and self._model is not None:
            if model_name != self._model_name:
                logger.warning(
                    "EmbeddingModel already loaded with %s. Ignoring model_name=%s",
                    self._model_name,
                    model_name,
                )
            return

        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._load_model(device=device, **kwargs)
        EmbeddingModel._initialized = True

    def _load_model(
        self,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """Load the sentence-transformers model."""
        try:
            self._model = SentenceTransformer(
                self._model_name,
                device=device,
                **kwargs,
            )
            logger.info("Loaded embedding model: %s", self._model_name)
        except OSError as e:
            logger.exception("Failed to load model %s (network/storage)", self._model_name)
            raise EmbeddingError(
                f"Could not load model '{self._model_name}'. "
                "Check network connection or use a cached model."
            ) from e
        except Exception as e:
            logger.exception("Unexpected error loading model %s", self._model_name)
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    @property
    def model(self) -> SentenceTransformer:
        """Access the underlying model. Raises if not loaded."""
        if self._model is None:
            raise EmbeddingError("Embedding model not loaded.")
        return self._model

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input text to embed.
            normalize: Whether to L2-normalize the embedding. Default True.

        Returns:
            1D numpy array of shape (embedding_dim,).

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not text or not isinstance(text, str):
            raise EmbeddingError("embed() requires a non-empty string")

        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return np.asarray(embedding, dtype=np.float32)
        except Exception as e:
            logger.exception("Failed to embed text (length=%d)", len(text))
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a batch of texts efficiently.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts per forward pass. Default 32.
            normalize: Whether to L2-normalize embeddings. Default True.
            show_progress: Whether to show a progress bar. Default False.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim).

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 0)

        if not isinstance(texts, (list, tuple)):
            raise EmbeddingError("embed_batch() requires a list or tuple of strings")

        invalid = [i for i, t in enumerate(texts) if not isinstance(t, str)]
        if invalid:
            raise EmbeddingError(
                f"All items must be strings. Invalid indices: {invalid[:5]}"
                + ("..." if len(invalid) > 5 else "")
            )

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            logger.exception("Failed to embed batch (size=%d)", len(texts))
            raise EmbeddingError(f"Batch embedding failed: {e}") from e

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton. Useful for testing or switching models."""
        cls._instance = None
        cls._initialized = False
        logger.debug("EmbeddingModel singleton reset")
