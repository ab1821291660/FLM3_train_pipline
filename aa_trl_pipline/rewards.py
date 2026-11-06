"""Picklable reward functions for synchronous and asynchronous GRPO."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


TOTAL_PATTERN = re.compile(r"TOTAL\s*=\s*([0-9]+(?:\.[0-9]{2})?)", re.IGNORECASE)


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk(nested)


def completion_text(completion: Any) -> str:
    """Extract assistant-facing text from standard or conversational outputs."""

    texts: list[str] = []
    for value in _walk(completion):##===================================
        if isinstance(value, dict) and isinstance(value.get("content"), str):
            if value.get("role") in {None, "assistant"}:
                texts.append(value["content"])
        elif isinstance(value, str):
            texts.append(value)
    return "\n".join(texts)

def called_tools(completion: Any) -> set[str]:
    """Collect function names from tool-call messages in a rollout transcript."""

    names: set[str] = set()
    for value in _walk(completion):##===================================
        if not isinstance(value, dict):
            continue
        function = value.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
        if value.get("role") == "tool" and isinstance(value.get("name"), str):
            names.add(value["name"])
    return names




def ________(self):pass
def exact_total_reward(
    completions: list[Any], expected_total: list[str], **kwargs: Any
) -> list[float]:
    """Reward an exact ``TOTAL=xx.xx`` final answer."""

    del kwargs
    rewards: list[float] = []
    for completion, expected in zip(completions, expected_total):
        matches = TOTAL_PATTERN.findall(completion_text(completion))##===================================
        rewards.append(float(bool(matches) and matches[-1] == str(expected)))##金额是否一致
    return rewards


def required_tools_reward(
    completions: list[Any], required_tools: list[list[str]], **kwargs: Any
) -> list[float]:
    """Reward rollouts whose transcript contains every required tool name."""

    del kwargs
    return [
        float(set(required).issubset(called_tools(completion)))##===================================##===================================
        for completion, required in zip(completions, required_tools)
    ]



def final_format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Reward a machine-checkable final answer regardless of its numeric value."""

    del kwargs
    return [float(bool(TOTAL_PATTERN.search(completion_text(item)))) for item in completions]##===================================

