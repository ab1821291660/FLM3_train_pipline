"""A five-stage, multi-turn agent post-training curriculum built on TRL."""

STAGE_ORDER = (
    "sft",
    "distillation",
    "grpo_tools",
    "grpo_openenv",
    "async_grpo",
)

__all__ = ["STAGE_ORDER"]
