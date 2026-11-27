# kb -> knowledge Base
import json
import re
from pathlib import Path
from typing import Any
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FAQ_PATH = BASE_DIR / "app" / "data" / "faq_ar.json"

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟـ]")
_NON_WORD = re.compile(r"[^\w\s]")


def normalize_arabic(text: str) -> str:
    """Light Arabic normalization so spelling variants match."""
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = _NON_WORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> set[str]:

    tokens = set()
    for token in normalize_arabic(text).split():
        tokens.add(token)
        # Strip the definite article so "العمل" also matches "عمل".
        if token.startswith("ال") and len(token) > 3:
            tokens.add(token[2:])
    return tokens


def _score(query_tokens: set[str], query_norm: str, entry: dict) -> float:

    score = 0.0
    for keyword in entry.get("keywords", []):
        keyword_norm = normalize_arabic(keyword)
        if keyword_norm and keyword_norm in query_norm:
            score += 2.0

    question_tokens = tokenize(entry["question"])
    score += len(query_tokens & question_tokens)

    return score


class FAQKnowledgeBase:

    def __init__(self, path: Path = FAQ_PATH):
        self.path = Path(path)
        self.entries: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"FAQ file not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError("Each FAQ entry must be an object")
            if not entry.get("question") or not entry.get("answer"):
                raise ValueError("Each FAQ entry requires question and answer")

        self.entries = data

    def search(self,query: str,top_k: int = 3,min_score: float = 1.0) -> list[dict[str, Any]]:

        if not query or not query.strip():
            return []

        query_norm = normalize_arabic(query)
        query_tokens = tokenize(query)

        scored = [
            (_score(query_tokens, query_norm, entry), entry)
            for entry in self.entries
        ]
        scored = [item for item in scored if item[0] >= min_score]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [entry for _, entry in scored[:top_k]]


    @staticmethod
    def format_results(results: list[dict[str, Any]]) -> str:

        if not results:
            return "No relevant FAQ entries found."

        return "\n\n".join(
            f"Q: {entry['question']}\nA: {entry['answer']}" for entry in results
        )


faq_kb = FAQKnowledgeBase()