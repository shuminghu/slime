"""
Standalone evaluation script for the Scientific Judge model.

Loads a trained model via vLLM/SGLang or transformers and evaluates on
SciJudgeBench test splits. Reports accuracy and compares to paper baselines.

Usage:
    python examples/scijudge/evaluate.py \
        --model-path /path/to/checkpoint \
        --data-dir /path/to/scijudge_data \
        --split test \
        --batch-size 32

Or evaluate the released HuggingFace model:
    python examples/scijudge/evaluate.py \
        --model-path OpenMOSS-Team/SciJudge-4B \
        --data-dir /path/to/scijudge_data \
        --split test
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_answer(response: str) -> str | None:
    """Extract the answer letter from model response."""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]

    match = re.search(r"<answer>\s*([ABab])\s*</answer>", response)
    if match:
        return match.group(1).upper()

    patterns = [
        r"(?:answer|choice)\s*(?:is|:)\s*([ABab])\b",
        r"(?:Paper\s+)([ABab])\s+(?:has|is|will)",
        r"final\s+answer\s*(?:is|:)\s*([ABab])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    candidates = re.findall(r"\b([ABab])\b", response)
    if candidates:
        return candidates[-1].upper()

    return None


def load_data(data_path: str) -> list[dict]:
    """Load JSONL evaluation data."""
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def evaluate_with_transformers(model_path: str, samples: list[dict], args):
    """Evaluate using HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    results = []
    for i, sample in enumerate(samples):
        messages = sample["input"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.temperature > 0,
        )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        predicted = extract_answer(response)
        correct = sample["label"]
        is_correct = predicted == correct if predicted else False

        results.append(
            {
                "index": i,
                "predicted": predicted,
                "correct": correct,
                "is_correct": is_correct,
                "response": response,
                "metadata": sample.get("metadata", {}),
            }
        )

        if (i + 1) % 50 == 0:
            acc = sum(r["is_correct"] for r in results) / len(results) * 100
            logger.info("Progress: %d/%d, running accuracy: %.1f%%", i + 1, len(samples), acc)

    return results


def evaluate_with_sglang(model_path: str, samples: list[dict], args):
    """Evaluate using SGLang for faster batch inference."""
    import sglang as sgl

    logger.info("Loading model from %s with SGLang", model_path)
    llm = sgl.Engine(model_path=model_path, tp_size=args.tp_size)

    prompts = []
    for sample in samples:
        messages = sample["input"]
        prompts.append(messages)

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    logger.info("Running batch inference on %d samples...", len(prompts))
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, (sample, output) in enumerate(zip(samples, outputs)):
        response = output["text"]
        predicted = extract_answer(response)
        correct = sample["label"]
        is_correct = predicted == correct if predicted else False

        results.append(
            {
                "index": i,
                "predicted": predicted,
                "correct": correct,
                "is_correct": is_correct,
                "response": response,
                "metadata": sample.get("metadata", {}),
            }
        )

    llm.shutdown()
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics, broken down by category."""
    total = len(results)
    correct = sum(r["is_correct"] for r in results)
    no_answer = sum(1 for r in results if r["predicted"] is None)

    metrics = {
        "overall_accuracy": correct / total * 100 if total > 0 else 0,
        "total_samples": total,
        "correct": correct,
        "no_answer_extracted": no_answer,
    }

    # Per-category breakdown
    by_category = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        cat = r.get("metadata", {}).get("paper_a_category", "Unknown")
        by_category[cat]["total"] += 1
        if r["is_correct"]:
            by_category[cat]["correct"] += 1

    metrics["per_category"] = {}
    for cat, counts in sorted(by_category.items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
        metrics["per_category"][cat] = {
            "accuracy": acc,
            "total": counts["total"],
            "correct": counts["correct"],
        }

    return metrics


# Paper baseline numbers for reference
PAPER_BASELINES = {
    "SciJudge-Qwen3-4B (paper)": {"in_domain": 77.1},
    "SciJudge-Qwen3-30B (paper)": {"in_domain": 80.6},
    "GPT-5.2 (paper)": {"in_domain": 75.7},
    "GLM-5 (paper)": {"in_domain": 69.2},
    "Gemini 3 Pro (paper)": {"in_domain": 72.5},
    "Random baseline": {"in_domain": 50.0},
}


def print_results(metrics: dict, split: str):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"  Scientific Judge Evaluation Results ({split})")
    print("=" * 70)
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.1f}%")
    print(f"  Total Samples:    {metrics['total_samples']}")
    print(f"  Correct:          {metrics['correct']}")
    print(f"  No Answer:        {metrics['no_answer_extracted']}")
    print()

    if metrics["per_category"]:
        print("  Per-Category Breakdown:")
        print("  " + "-" * 50)
        for cat, cat_metrics in metrics["per_category"].items():
            print(f"    {cat:30s} {cat_metrics['accuracy']:5.1f}%  ({cat_metrics['correct']}/{cat_metrics['total']})")
        print()

    print("  Paper Baselines (in-domain):")
    print("  " + "-" * 50)
    for name, baseline in PAPER_BASELINES.items():
        print(f"    {name:35s} {baseline['in_domain']:5.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scientific Judge model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint or HF model ID")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing JSONL data files")
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "test_ood_iclr", "test_ood_year"], help="Eval split"
    )
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "sglang"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for SGLang")
    parser.add_argument("--output-path", type=str, default=None, help="Save detailed results to JSONL")
    args = parser.parse_args()

    data_path = f"{args.data_dir}/{args.split}.jsonl"
    logger.info("Loading evaluation data from %s", data_path)
    samples = load_data(data_path)
    logger.info("Loaded %d samples", len(samples))

    if args.backend == "sglang":
        results = evaluate_with_sglang(args.model_path, samples, args)
    else:
        results = evaluate_with_transformers(args.model_path, samples, args)

    metrics = compute_metrics(results)
    print_results(metrics, args.split)

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Detailed results saved to %s", args.output_path)

    # Return non-zero exit if accuracy is below random baseline
    if metrics["overall_accuracy"] < 50.0:
        logger.warning("Accuracy below random baseline (50%%)!")
        sys.exit(1)


if __name__ == "__main__":
    main()
