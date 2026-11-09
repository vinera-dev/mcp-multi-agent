import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FAQ_DIR = Path(__file__).parent / "data" / "faq"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
)
K1 = 1.5
B = 0.75


@dataclass(frozen=True)
class FaqEntry:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class FaqHit:
    id: str
    title: str
    text: str
    score: float


SUFFIXES = ("ations", "ation", "ings", "ing", "ed", "es", "s")
MIN_STEM_LENGTH = 4


def stem(token: str) -> str:
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_LENGTH:
            token = token[: -len(suffix)]
            break
    if len(token) > MIN_STEM_LENGTH and token.endswith("e"):
        token = token[:-1]
    if len(token) > MIN_STEM_LENGTH and token[-1] == token[-2] and token[-1] not in "aeiou":
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(token) for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


def load_entries(directory: Path = DEFAULT_FAQ_DIR) -> list[FaqEntry]:
    entries = []
    for path in sorted(directory.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip()
        text = " ".join(line.strip() for line in lines[1:] if line.strip())
        entries.append(FaqEntry(id=path.stem, title=title, text=text))
    return entries


class FaqIndex:
    def __init__(self, entries: list[FaqEntry]) -> None:
        self._entries = entries
        self._tokens = [tokenize(f"{entry.title} {entry.text}") for entry in entries]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        total = len(entries)
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[FaqHit]:
        terms = set(tokenize(query))
        scored = []
        for index, entry in enumerate(self._entries):
            score = self._score(terms, index)
            if score > 0:
                scored.append(FaqHit(entry.id, entry.title, entry.text, round(score, 4)))
        scored.sort(key=lambda hit: (-hit.score, hit.id))
        return scored[:top_k]

    def _score(self, terms: set[str], index: int) -> float:
        frequencies = self._frequencies[index]
        length_norm = 1 - B + B * self._lengths[index] / self._average
        total = 0.0
        for term in terms:
            count = frequencies.get(term, 0)
            if count:
                total += self._idf[term] * count * (K1 + 1) / (count + K1 * length_norm)
        return total
