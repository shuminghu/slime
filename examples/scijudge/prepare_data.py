"""
Prepare SciJudgeBench data for slime training.

Downloads the OpenMOSS-Team/SciJudgeBench dataset from HuggingFace and converts
it to the JSONL format expected by slime.

Each output row has:
  - "input": list of chat messages (system + user) with the paper comparison prompt
  - "label": correct answer ("A" or "B")
  - "metadata": additional fields for analysis (arxiv IDs, citations, categories)

Usage:
    python examples/scijudge/prepare_data.py --output-dir /path/to/output
"""

import argparse
import json
import logging
import os

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

USER_TEMPLATE = (
    "Today is {date}. Based on the titles, abstracts, and publication dates of the "
    "following two papers A and B, determine which paper has a higher citation count.\n"
    "Show your reasoning process in <reason> </reason> tags. And return the final "
    "answer in <answer> </answer> tags. The final answer should contain only 'A' or 'B'.\n\n"
    "Paper A:\nTitle: {title_a}\nAbstract: {abstract_a}\nDate: {date_a}\n\n"
    "Paper B:\nTitle: {title_b}\nAbstract: {abstract_b}\nDate: {date_b}"
)

# Reference date used by the paper's model card
DEFAULT_DATE = "2025-12-10"


def format_date(ts) -> str:
    """Format a timestamp or string to YYYY-MM-DD."""
    if ts is None:
        return "Unknown"
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d")
    return str(ts)[:10]


def convert_row(row: dict, reference_date: str = DEFAULT_DATE) -> dict | None:
    """Convert a single HuggingFace dataset row to slime training format."""
    # If the dataset already has pre-built messages, extract info from them
    correct_answer = row.get("correct_answer")
    if not correct_answer:
        return None

    title_a = row.get("paper_a_title", "")
    title_b = row.get("paper_b_title", "")
    abstract_a = row.get("paper_a_abstract", "")
    abstract_b = row.get("paper_b_abstract", "")
    date_a = format_date(row.get("paper_a_date"))
    date_b = format_date(row.get("paper_b_date"))

    if not title_a or not title_b:
        return None

    user_content = USER_TEMPLATE.format(
        date=reference_date,
        title_a=title_a,
        abstract_a=abstract_a or "N/A",
        date_a=date_a,
        title_b=title_b,
        abstract_b=abstract_b or "N/A",
        date_b=date_b,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    metadata = {
        "paper_a_arxiv_id": row.get("paper_a_arxiv_id", ""),
        "paper_b_arxiv_id": row.get("paper_b_arxiv_id", ""),
        "paper_a_citations": row.get("paper_a_citations"),
        "paper_b_citations": row.get("paper_b_citations"),
        "paper_a_category": row.get("paper_a_category", ""),
        "paper_b_category": row.get("paper_b_category", ""),
    }

    return {
        "input": messages,
        "label": correct_answer.strip().upper(),
        "metadata": metadata,
    }


def write_jsonl(rows, path):
    """Write rows to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            converted = convert_row(row)
            if converted is not None:
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare SciJudgeBench data for slime")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for JSONL files")
    parser.add_argument("--reference-date", type=str, default=DEFAULT_DATE, help="Reference date for prompts")
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Max training samples (None = use all)"
    )
    args = parser.parse_args()

    logger.info("Loading SciJudgeBench from HuggingFace...")
    ds = load_dataset("OpenMOSS-Team/SciJudgeBench")
    logger.info("Available splits: %s", list(ds.keys()))

    for split_name in ds:
        split_data = ds[split_name]
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")

        if split_name == "train" and args.max_train_samples:
            split_data = split_data.select(range(min(args.max_train_samples, len(split_data))))

        logger.info("Converting %s (%d rows) -> %s", split_name, len(split_data), out_path)
        count = write_jsonl(split_data, out_path)
        logger.info("Wrote %d samples to %s", count, out_path)

    logger.info("Done! Data saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
