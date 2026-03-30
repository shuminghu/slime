"""
Custom reward function for Scientific Judge training.

The model is trained to predict which of two papers has higher citation count.
It outputs reasoning in <reason></reason> tags and the final answer (A or B)
in <answer></answer> tags.

Reward: 1.0 if the extracted answer matches the ground truth label, 0.0 otherwise.
"""

import re

from slime.utils.types import Sample


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks if present."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def _extract_answer(response: str) -> str | None:
    """Extract the answer letter from <answer>...</answer> tags."""
    text = _strip_thinking(response)

    # Try <answer> tags first
    match = re.search(r"<answer>\s*([ABab])\s*</answer>", text)
    if match:
        return match.group(1).upper()

    # Fallback: look for common answer patterns
    patterns = [
        r"(?:answer|choice)\s*(?:is|:)\s*([ABab])\b",
        r"(?:Paper\s+)([ABab])\s+(?:has|is|will)",
        r"final\s+answer\s*(?:is|:)\s*([ABab])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: last standalone A or B
    candidates = re.findall(r"\b([ABab])\b", text)
    if candidates:
        return candidates[-1].upper()

    return None


async def custom_rm(args, sample: Sample, **kwargs) -> float:
    """Compute reward for a single sample.

    Returns 1.0 if the model's answer matches the correct answer, 0.0 otherwise.
    """
    response = sample.response
    label = sample.label  # "A" or "B"

    if not response or not label:
        return 0.0

    extracted = _extract_answer(response)
    if extracted is None:
        return 0.0

    return 1.0 if extracted == label.strip().upper() else 0.0
