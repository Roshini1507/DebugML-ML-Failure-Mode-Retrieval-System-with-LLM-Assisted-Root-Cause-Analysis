"""Production-grade RAG pipeline for ML failure debugging."""

from __future__ import annotations

import json
import logging
from typing import TypedDict

from llm import OllamaLLM, OllamaLLMError
from vector_db import SearchResult, VectorDB, VectorDBError

logger = logging.getLogger(__name__)


class RAGResponse(TypedDict):
    """Structured JSON output from the RAG pipeline."""

    root_cause: str
    confidence: str
    recommended_solution: str
    similar_failures: list[SearchResult]


class RAGError(Exception):
    """Raised when RAG pipeline operations fail."""

    pass


class RAGPipeline:
    """
    Production-grade RAG pipeline for ML failure root cause analysis.

    Pipeline steps:
    1. Embed user query (internal to VectorDB)
    2. Retrieve similar failures from FAISS
    3. Construct context from retrieved failures
    4. Send context and query to Ollama LLM
    5. Return structured output (similar failures + LLM explanation)
    """

    def __init__(
        self,
        vector_db: VectorDB | None = None,
        llm: OllamaLLM | None = None,
        top_k: int = 3,
    ) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            vector_db: VectorDB instance. If None, creates default.
            llm: OllamaLLM instance. If None, creates default.
            top_k: Number of similar failures to retrieve. Default 3.
        """
        self._vector_db = vector_db or VectorDB()
        self._llm = llm or OllamaLLM()
        self._top_k = top_k

    def query(self, user_query: str, top_k: int | None = None) -> RAGResponse:
        """
        Run the full RAG pipeline and return structured output.

        Steps:
        1. Embed user query (internal to retrieval)
        2. Retrieve similar failures from FAISS vector store
        3. Construct context from retrieved failures
        4. Send context and query to Ollama LLM
        5. Return similar failures + LLM explanation

        Args:
            user_query: User's ML problem or symptom description.
            top_k: Override default number of results. If None, uses instance default.

        Returns:
            RAGResponse (structured JSON) with root_cause, confidence,
            recommended_solution, and similar_failures.

        Raises:
            RAGError: If pipeline fails.
        """
        k = top_k if top_k is not None else self._top_k

        if not user_query or not isinstance(user_query, str):
            raise RAGError("query() requires a non-empty string")

        logger.info("RAG query: retrieving top-%d failures", k)

        try:
            # Step 1: Embed query (internal) + Step 2: Retrieve similar failures
            similar_failures = self._vector_db.search(user_query, k=k)

            logger.info("Retrieved %d similar failures", len(similar_failures))

            # Step 3 & 4: Construct context and send to LLM (structured JSON output)
            analysis = self._llm.generate_structured_response(
                context=similar_failures,
                query=user_query,
            )

            # Step 5: Return structured JSON output
            return RAGResponse(
                root_cause=analysis["root_cause"],
                confidence=analysis["confidence"],
                recommended_solution=analysis["recommended_solution"],
                similar_failures=similar_failures,
            )

        except VectorDBError as e:
            logger.exception("Vector DB retrieval failed")
            raise RAGError(f"Retrieval failed: {e}") from e
        except OllamaLLMError as e:
            logger.exception("LLM generation failed")
            raise RAGError(f"Generation failed: {e}") from e


def to_json(response: RAGResponse) -> str:
    """Serialize RAGResponse to JSON string."""
    return json.dumps(response, indent=2, default=str)
