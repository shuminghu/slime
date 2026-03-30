"""Tests for the Scientific Judge reward function."""

import asyncio
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.scijudge.reward import _extract_answer, custom_rm
from slime.utils.types import Sample


class TestExtractAnswer:
    """Test answer extraction from model responses."""

    def test_answer_tag_a(self):
        assert _extract_answer("<reason>Paper A is better...</reason><answer>A</answer>") == "A"

    def test_answer_tag_b(self):
        assert _extract_answer("<reason>Paper B has more impact</reason><answer>B</answer>") == "B"

    def test_answer_tag_with_whitespace(self):
        assert _extract_answer("<answer> A </answer>") == "A"

    def test_answer_tag_lowercase(self):
        assert _extract_answer("<answer>b</answer>") == "B"

    def test_thinking_block_stripped(self):
        response = "<think>Let me reason about this...</think><reason>Analysis</reason><answer>A</answer>"
        assert _extract_answer(response) == "A"

    def test_fallback_answer_is_pattern(self):
        assert _extract_answer("After analysis, the answer is A.") == "A"

    def test_fallback_choice_is(self):
        assert _extract_answer("My choice is B based on citations.") == "B"

    def test_fallback_paper_pattern(self):
        assert _extract_answer("Paper B has higher impact potential.") == "B"

    def test_fallback_last_standalone_letter(self):
        assert _extract_answer("Comparing the two, I think B") == "B"

    def test_no_answer_found(self):
        assert _extract_answer("This response discusses the methodology of both studies.") is None

    def test_empty_response(self):
        assert _extract_answer("") is None

    def test_multiple_tags_uses_first(self):
        # regex finds first match
        assert _extract_answer("<answer>A</answer> some text <answer>B</answer>") == "A"

    def test_ignores_non_ab_letters(self):
        response = "The answer is C, but actually the answer is A."
        assert _extract_answer(response) == "A"


class TestCustomRM:
    """Test the full reward function."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_sample(self, response="", label="A"):
        return Sample(response=response, label=label)

    def test_correct_answer_a(self):
        sample = self._make_sample("<answer>A</answer>", "A")
        assert self._run(custom_rm(None, sample)) == 1.0

    def test_correct_answer_b(self):
        sample = self._make_sample("<answer>B</answer>", "B")
        assert self._run(custom_rm(None, sample)) == 1.0

    def test_wrong_answer(self):
        sample = self._make_sample("<answer>A</answer>", "B")
        assert self._run(custom_rm(None, sample)) == 0.0

    def test_no_response(self):
        sample = self._make_sample("", "A")
        assert self._run(custom_rm(None, sample)) == 0.0

    def test_no_label(self):
        sample = self._make_sample("<answer>A</answer>", None)
        assert self._run(custom_rm(None, sample)) == 0.0

    def test_no_extractable_answer(self):
        sample = self._make_sample("I cannot determine the answer.", "A")
        assert self._run(custom_rm(None, sample)) == 0.0

    def test_label_whitespace(self):
        sample = self._make_sample("<answer>B</answer>", " B ")
        assert self._run(custom_rm(None, sample)) == 1.0

    def test_label_lowercase(self):
        sample = self._make_sample("<answer>A</answer>", "a")
        assert self._run(custom_rm(None, sample)) == 1.0

    def test_full_reasoning_response(self):
        response = (
            "<think>Let me analyze both papers carefully.</think>"
            "<reason>Paper A focuses on a niche topic with limited citations. "
            "Paper B introduces a novel method that has been widely adopted.</reason>"
            "<answer>B</answer>"
        )
        sample = self._make_sample(response, "B")
        assert self._run(custom_rm(None, sample)) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
