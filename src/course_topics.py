from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional


TOPIC_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "direct methods": (
        "lu",
        "gaussian elimination",
        "pivot",
        "triangular",
        "cholesky",
        "direct method",
    ),
    "sparse and banded systems": (
        "sparse",
        "banded",
        "tridiagonal",
        "storage",
        "fill-in",
    ),
    "iterative methods": (
        "jacobi",
        "gauss-seidel",
        "sor",
        "successive over-relaxation",
        "spectral radius",
    ),
    "conjugate gradient": (
        "conjugate gradient",
        "cg",
        "positive definite",
        "spd",
        "krylov",
    ),
    "qr factorization": (
        "qr",
        "householder",
        "givens",
        "orthogonal factorization",
    ),
    "svd": (
        "svd",
        "singular value",
        "rank",
        "condition number",
        "low rank",
    ),
    "least squares": (
        "least squares",
        "overdetermined",
        "normal equations",
        "projection",
        "orthogonal projection",
    ),
    "eigenvalue methods": (
        "eigenvalue",
        "eigenvector",
        "power method",
        "inverse iteration",
        "qr algorithm",
    ),
}


def infer_topic(text: str, fallback: str = "general") -> str:
    lowered = text.lower()
    scores = {
        topic: sum(1 for keyword in keywords if keyword in lowered)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else fallback


def infer_topic_from_path(file_path: Path) -> str:
    path_text = " ".join(file_path.with_suffix("").parts)
    return infer_topic(path_text, fallback="general")


def expand_query_with_topic_terms(question: str, topic: Optional[str] = None) -> str:
    terms: Iterable[str] = TOPIC_KEYWORDS.get(topic or "", ())
    if not terms:
        inferred_topic = infer_topic(question)
        terms = TOPIC_KEYWORDS.get(inferred_topic, ())
    extras = ", ".join(list(terms)[:5])
    return f"{question.strip()} {extras}".strip()
