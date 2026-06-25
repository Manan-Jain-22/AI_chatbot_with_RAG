from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from src.config import BASE_DIR, DEFAULT_TOP_K
from src.course_topics import expand_query_with_topic_terms, infer_topic
from src.vector_store import get_retriever


EVAL_DIR = BASE_DIR / "eval"
DEFAULT_EVAL_FILE = EVAL_DIR / "golden_questions.csv"
DEFAULT_RESULTS_FILE = EVAL_DIR / "results.csv"


def _matches_expected(row: dict, document) -> bool:
    expected_topic = row.get("expected_topic", "").strip().lower()
    expected_section = row.get("expected_section", "").strip().lower()
    expected_source = row.get("expected_source", "").strip().lower()

    metadata = document.metadata
    topic = str(metadata.get("topic", "")).lower()
    section = str(metadata.get("section_heading", "")).lower()
    source = str(metadata.get("source", "")).lower()

    topic_match = not expected_topic or expected_topic == topic
    section_match = not expected_section or expected_section in section
    source_match = not expected_source or expected_source in source
    return topic_match and section_match and source_match


def run_retrieval_eval(
    rows: Iterable[dict],
    output_path: Path = DEFAULT_RESULTS_FILE,
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """Evaluate retrieval against a small manually labeled golden set."""
    retriever = get_retriever(top_k=top_k)
    results = []

    for row in rows:
        question = row["question"]
        inferred_topic = infer_topic(question)
        query = expand_query_with_topic_terms(question, inferred_topic)
        documents = retriever.invoke(query)
        hits = [document for document in documents if _matches_expected(row, document)]
        precision = len(hits) / len(documents) if documents else 0
        recall = 1.0 if hits else 0.0
        retrieved_sources = [
            f"{doc.metadata.get('source', '')}::{doc.metadata.get('section_heading', '')}"
            for doc in documents
        ]

        results.append(
            {
                "question": question,
                "expected_topic": row.get("expected_topic", ""),
                "expected_section": row.get("expected_section", ""),
                "inferred_topic": inferred_topic,
                f"precision_at_{top_k}": f"{precision:.3f}",
                f"recall_at_{top_k}": f"{recall:.3f}",
                "retrieved_sources": "; ".join(retrieved_sources),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question",
                "expected_topic",
                "expected_section",
                "inferred_topic",
                f"precision_at_{top_k}",
                f"recall_at_{top_k}",
                "retrieved_sources",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return results


def load_golden_questions(path: Path = DEFAULT_EVAL_FILE) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}. Create a CSV with question, "
            "expected_topic, expected_section, expected_source, and key_answer_points columns."
        )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "question" not in reader.fieldnames:
            raise ValueError("Evaluation CSV must include a 'question' column.")
        return [row for row in reader if row.get("question", "").strip()]


if __name__ == "__main__":
    eval_questions = load_golden_questions()
    results = run_retrieval_eval(eval_questions)
    print(f"Wrote {len(results)} evaluation rows to {DEFAULT_RESULTS_FILE}")
