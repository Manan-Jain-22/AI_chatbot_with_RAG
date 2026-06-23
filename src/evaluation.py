from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from src.config import BASE_DIR, DEFAULT_TOP_K
from src.rag_chain import answer_question


EVAL_DIR = BASE_DIR / "eval"
DEFAULT_EVAL_FILE = EVAL_DIR / "questions.csv"
DEFAULT_RESULTS_FILE = EVAL_DIR / "results.csv"


def run_smoke_eval(
    questions: Iterable[str],
    output_path: Path = DEFAULT_RESULTS_FILE,
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """Run a lightweight retrieval/generation evaluation over plain questions."""
    rows = []
    for question in questions:
        result = answer_question(question, top_k=top_k)
        rows.append(
            {
                "question": question,
                "rewritten_question": result["rewritten_question"],
                "answer": result["answer"],
                "sources": "; ".join(result["sources"]),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question", "rewritten_question", "answer", "sources"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def load_questions(path: Path = DEFAULT_EVAL_FILE) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation questions file not found: {path}. Create a CSV with a question column."
        )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "question" not in reader.fieldnames:
            raise ValueError("Evaluation CSV must include a 'question' column.")
        return [row["question"].strip() for row in reader if row.get("question", "").strip()]


if __name__ == "__main__":
    eval_questions = load_questions()
    results = run_smoke_eval(eval_questions)
    print(f"Wrote {len(results)} evaluation rows to {DEFAULT_RESULTS_FILE}")
