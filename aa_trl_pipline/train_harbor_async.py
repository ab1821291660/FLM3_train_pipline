"""Optional black-box AsyncGRPO path for a real agent harness via OpenEnv Harbor.

This is deliberately separate from the five local datasets: a Harbor server
owns its task dataset, sandbox image, verifier, and captured token traces.  Use
``--dry-run`` to inspect the topology without importing OpenEnv, TRL, or vLLM.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any


def harness_reward(outcome: Any) -> float | None:
    """Use verifier correctness and a small successful-tool-efficiency bonus."""

    if outcome.env_reward is None:
        return None
    correctness = float(outcome.env_reward)
    calls = outcome.tool_call_count
    if correctness >= 1.0 and calls is not None:
        correctness += 0.2 * max(0.0, min(1.0, 1.0 - float(calls) / 15.0))
    return correctness


def plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "dry-run: no imports, server calls, model loads, or network access",
        "harbor_server": args.server,
        "vllm_server": args.vllm_url,
        "model": args.model,
        "task_dataset": args.split,
        "harness": args.harness,
        "sandbox": args.sandbox,
        "topology": [
            "OpenEnv Harbor owns tasks and isolated sandboxes",
            "the real harness owns the multi-turn tool loop",
            "vLLM capture returns token ids, logprobs, and loss masks",
            "AsyncGRPOTrainer consumes traces and synchronizes updated weights",
        ],
    }


def train(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from harbor_env.harness import HarborSessionFactory
    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
    from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker

    factory = partial(
        HarborSessionFactory,
        args.server,
        split=args.split,
        harness=args.harness,
        sandbox=args.sandbox,
        llm_url=args.vllm_url,
        model=args.model,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        num_tasks=args.n_tasks,
    )
    metadata_factory = factory()
    try:
        dataset = Dataset.from_list(metadata_factory.prompt_rows())
    finally:
        internal_env = getattr(metadata_factory, "_env", None)
        if internal_env is not None:
            internal_env.close()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AsyncGRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        vllm_server_base_url=args.vllm_url,
        dtype="bfloat16",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=max(1, args.max_steps),
        report_to="none",
        log_completions=True,
    )
    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        harness_adapter=None,
        rollout_reward_fn=harness_reward,
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],
        processing_class=tokenizer,
        chat_template_kwargs={"enable_thinking": False},
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        log_completions=True,
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server", default="http://127.0.0.1:8200")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--split", default="<harbor-task-dataset>")
    parser.add_argument("--harness", default="mini-swe-agent")
    parser.add_argument("--sandbox", default="e2b")
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--agent-timeout", type=float, default=300.0)
    parser.add_argument("--agent-step-limit", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/harbor_async"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
    else:
        train(args)


if __name__ == "__main__":
    main()
