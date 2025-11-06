"""Utilities for Vietnamese word suggestions based on partial text."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


@dataclass
class VietnameseWordSuggester:
    """Provide simple prefix-based Vietnamese word suggestions."""

    words: List[str]

    @classmethod
    def from_default(cls) -> "VietnameseWordSuggester":
        data_path = Path(__file__).with_name("data").joinpath("vietnamese_words.txt")
        if not data_path.exists():
            raise FileNotFoundError(f"Missing default vocabulary at {data_path}")
        with data_path.open(encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return cls(words=list(words))

    def _normalise(self, value: str) -> str:
        return _strip_accents(value).replace(" ", "").lower()

    def _extract_prefix(self, text: str) -> str:
        text = text.rstrip()
        if not text:
            return ""
        if text.endswith(" "):
            return ""
        return text.split(" ")[-1]

    def suggest(self, text: str, limit: int = 3) -> List[str]:
        prefix = self._extract_prefix(text)
        if not prefix:
            return []
        norm_prefix = self._normalise(prefix)
        candidates: List[str] = []
        for word in self.words:
            if self._normalise(word).startswith(norm_prefix):
                candidates.append(word)
            if len(candidates) >= limit:
                break
        return candidates

    def best_suggestion(self, text: str) -> str:
        suggestions = self.suggest(text, limit=1)
        return suggestions[0] if suggestions else ""

    def apply_suggestion(self, text: str, suggestion: str) -> str:
        text = text.rstrip()
        if not text:
            return suggestion + " "
        if text.endswith(" "):
            return text + suggestion + " "
        base = text.split(" ")[:-1]
        updated_words = base + [suggestion, ""]
        return " ".join(updated_words).strip() + " "

    def extend_vocabulary(self, words: Iterable[str]) -> None:
        for word in words:
            cleaned = word.strip()
            if cleaned and cleaned not in self.words:
                self.words.append(cleaned)
