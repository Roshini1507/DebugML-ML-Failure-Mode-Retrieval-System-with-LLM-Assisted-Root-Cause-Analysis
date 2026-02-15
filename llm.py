"""Local LLM module for Ollama - root cause analysis and solution generation."""

from __future__ import annotations

import json
import logging
import re

import ollama
from pydantic import BaseModel

from vector_db import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3:latest"

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string", "description": "Likely cause(s) of the failure"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"], "description": "Confidence in the analysis"},
        "recommended_solution": {"type": "string", "description": "Concrete, actionable steps to fix the issue"},
    },
    "required": ["root_cause", "confidence", "recommended_solution"],
}

_SYSTEM_PROMPT = """You are an expert ML engineer helping debug model failures. Given similar past failures and a user's problem description, respond with a JSON object containing:
- root_cause: Likely cause(s) based on the symptoms and similar cases
- confidence: "high", "medium", or "low" based on how well the problem matches the examples
- recommended_solution: Concrete, actionable steps to fix the issue

Be concise, practical, and cite relevant patterns from the provided examples. Output only valid JSON."""


class OllamaLLMError(Exception):
    """Raised when Ollama LLM operations fail."""

    pass


def _format_context(search_results: list[SearchResult]) -> str:
    """Format retrieved failures into context text for the LLM."""
    if not search_results:
        return "No similar past failures found."

    blocks = []
    for r in search_results:
        f = r.get("failure", {})
        score = r.get("score", 0)
        blocks.append(
            f"""---
Similar failure (relevance: {score:.2f}):
- Description: {f.get('failure_description', '')}
- Root cause: {f.get('root_cause', '')}
- Solution: {f.get('solution', '')}
- Tags: {', '.join(f.get('tags', []))}"""
        )
    return "\n\n".join(blocks)


def normalize_llm_field(value: str | list | None) -> str:
    """
    Normalize an LLM response field to a string.
    - list → string using ". ".join()
    - None → empty string
    - everything else → string
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ". ".join(str(x) for x in value)
    return str(value)


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, with fallback for embedded JSON."""
    content = content.strip()
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try to extract JSON block
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise OllamaLLMError("LLM response could not be parsed as JSON")


class LLMAnalysis(BaseModel):
    """Structured output from LLM analysis."""

    root_cause: str
    confidence: str
    recommended_solution: str

    def to_dict(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "recommended_solution": self.recommended_solution,
        }


class OllamaLLM:
    """
    Local LLM  using Ollama (no API keys required).
    Uses llama3 for root cause analysis and solution generation.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the Ollama LLM.

        Args:
            model: Ollama model name. Default: llama3
            host: Ollama server URL. Default: localhost:11434
        """
        self.model = model


    def generate_structured_response(
        self,
        context: list[SearchResult],
        query: str,
    ) -> dict:
        """
        Generate structured JSON output: root_cause, confidence, recommended_solution.

        Args:
            context: List of SearchResult from vector DB (similar past failures).
            query: User's ML problem or symptom description.

        Returns:
            Dict with keys: root_cause, confidence, recommended_solution.

        Raises:
            OllamaLLMError: If generation or parsing fails.
        """
        if not query or not isinstance(query, str):
            raise OllamaLLMError("generate_structured_response() requires a non-empty string query")

        context_text = _format_context(context)
        user_message = f"""Similar past failures:

{context_text}

---

User's problem:
{query}

Respond with a JSON object: root_cause, confidence ("high"|"medium"|"low"), recommended_solution."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                format="json",
            )
            content = response["message"]["content"]
            if content is None:
                raise OllamaLLMError("Ollama returned empty response")

            parsed = _parse_json_response(content)
            root_cause = normalize_llm_field(parsed.get("root_cause"))
            recommended_solution = normalize_llm_field(parsed.get("recommended_solution"))
            confidence_raw = normalize_llm_field(parsed.get("confidence")) or "medium"
            confidence = (
                confidence_raw.lower()
                if confidence_raw.lower() in ("high", "medium", "low")
                else "medium"
            )
            analysis = LLMAnalysis(
                root_cause=root_cause,
                confidence=confidence,
                recommended_solution=recommended_solution,
            )
            return analysis.to_dict()
        except OllamaLLMError:
            raise
        except Exception as e:

            print("\n\n========== REAL OLLAMA ERROR ==========")
            print(type(e))
            print(str(e))

            import traceback
            traceback.print_exc()

            print("=======================================\n\n")

            raise OllamaLLMError(str(e))

