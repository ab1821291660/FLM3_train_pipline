"""Run the five-stage TRL multi-turn Agent RL curriculum.

Heavy ML imports are lazy.  ``--dry-run`` validates all datasets and prints the
handoff plan without importing TRL, loading checkpoints, or accessing a network.
Actual remote checkpoint loading is blocked unless ``--allow-model-download``
is supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aa_trl_pipline import STAGE_ORDER
from aa_trl_pipline.order_env import OrderFulfillmentEnv
from aa_trl_pipline.prepare_data import DATA_DIR, STAGE_FILES, load_jsonl, validate_datasets
from aa_trl_pipline.rewards import exact_total_reward, final_format_reward, required_tools_reward
from aa_trl_pipline.toolkit import TOOLS


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
TRAINERS = {
    "sft": "SFTTrainer",
    "distillation": "DistillationTrainer",
    "grpo_tools": "GRPOTrainer(tools=...)",
    "grpo_openenv": "GRPOTrainer(environment_factory=...)",
    "async_grpo": "experimental.AsyncGRPOTrainer(environment_factory=...)",
}
WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")
def has_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(next(path.glob(pattern), None) is not None for pattern in WEIGHT_PATTERNS)
# resolve_model_source(args.student_model, args.allow_model_download, "student_model")
def resolve_model_source(value: str, allow_model_download: bool, label: str) -> str:
    """Return a local checkpoint or an explicitly authorized Hub model ID."""
    path = Path(value).expanduser()
    if path.exists():
        resolved = path.resolve()
        if not has_checkpoint(resolved):
            raise FileNotFoundError(f"{label} path has no checkpoint weights: {resolved}")
        return str(resolved)
    if not allow_model_download:
        raise RuntimeError(
            f"{label}={value!r} is not a local checkpoint. Refusing model download; "
            "provide a local path or pass --allow-model-download explicitly."
        )
    return value


def selected_stages(values: list[str]) -> tuple[str, ...]:
    return STAGE_ORDER if values == ["all"] else tuple(values)


def ____辅助函数(self):pass
def stage_payload(stage: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only columns consumed by the corresponding trainer or environment."""
    if stage == "sft":##===================================
        return [{"messages": row["messages"], "tools": row["tools"]} for row in rows]
    if stage == "distillation":##===================================
        return [{"prompt": row["prompt"]} for row in rows]
    if stage == "grpo_tools":##===================================
        return [
            {
                "prompt": row["prompt"],
                "expected_total": row["expected_total"],##总值
                "required_tools": row["required_tools"],
            }
            for row in rows
        ]
    if stage in {"grpo_openenv",##===================================
                 "async_grpo"}:##===================================
        columns = ("prompt", "sku_a", "qty_a", "sku_b", "qty_b", "city", "expected_total")
        return [{column: row[column] for column in columns} for row in rows]
    raise ValueError(stage)





def _dataset_from_rows(rows: list[dict[str, Any]], mixed_json: bool = False) -> Any:
    from datasets import Dataset

    if mixed_json:
        try:
            return Dataset.from_list(rows, on_mixed_types="use_json")
        except TypeError:
            fallback = [dict(row, tools=json.dumps(row["tools"], ensure_ascii=False)) for row in rows]
            return Dataset.from_list(fallback)
    return Dataset.from_list(rows)


def _save(trainer: Any, tokenizer: Any, output_dir: Path) -> None:
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

def _common_kwargs(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    steps = 1 if args.smoke  else args.max_steps
    return {
        "output_dir": str(args.output_dir / stage),
        "max_steps": steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "save_strategy": "steps",
        "save_steps": max(1, steps),
        "report_to": "none",
        "logging_steps": 1,
        "seed": args.seed,
    }



def ____sft类(self):pass
def _train_sft(args: argparse.Namespace, source: str, rows: list[dict[str, Any]]) -> str:
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    output = args.output_dir / "sft"
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    dataset = _dataset_from_rows(stage_payload("sft", rows), mixed_json=True)##===================================
    trainer = SFTTrainer(
        model=source,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            **_common_kwargs(args, "sft"),##===================================
            max_length=args.max_length,
            assistant_only_loss=True,##===================================
        ),
    )
    _save(trainer, tokenizer, output)##===================================
    return str(output)


def ____蒸馏soft白盒(self):pass
####白盒蒸馏（White-box Distillation） 拥有对教师模型的“完全访问权”，不仅能获得其最终输出，还能深入其内部，利用隐藏状态、注意力矩阵等中间特征进行学习。
def _train_distillation(
    args: argparse.Namespace, source: str, teacher: str, rows: list[dict[str, Any]]
) -> str:
    from transformers import AutoTokenizer
    from trl import DistillationConfig, DistillationTrainer

    output = args.output_dir / "distillation"
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    dataset = _dataset_from_rows(stage_payload("distillation", rows))##===================================
    trainer = DistillationTrainer(
        model=source,
        teacher_model=teacher,
        processing_class=tokenizer,
        train_dataset=dataset,
        tools=TOOLS,##===================================##===================================##===================================
        args=DistillationConfig(
            **_common_kwargs(args, "distillation"),
            max_completion_length=args.max_completion_length,
            max_tool_calling_iterations=args.max_tool_calling_iterations,##8次最多 ##===================================##===================================##===================================
        ),
    )
    _save(trainer, tokenizer, output)
    return str(output)


def ____rl类(self):pass
def _grpo_config(args: argparse.Namespace, stage: str) -> Any:
    from trl import GRPOConfig

    return GRPOConfig(
        **_common_kwargs(args, stage),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    )


def _environment_factory(args: argparse.Namespace) -> Any:
    if not args.openenv_url:
        return OrderFulfillmentEnv
    from aa_trl_pipline.openenv_adapter import OpenEnvOrderFactory

    return OpenEnvOrderFactory(args.openenv_url)


def _train_grpo_tools(args: argparse.Namespace, source: str, rows: list[dict[str, Any]]) -> str:
    from transformers import AutoTokenizer
    from trl import GRPOTrainer

    output = args.output_dir / "grpo_tools"
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    dataset = _dataset_from_rows(stage_payload("grpo_tools", rows))
    trainer = GRPOTrainer(
        model=source,
        processing_class=tokenizer,
        train_dataset=dataset,
        tools=TOOLS,
        reward_funcs=[exact_total_reward, required_tools_reward],
        args=_grpo_config(args, "grpo_tools"),
    )
    _save(trainer, tokenizer, output)
    return str(output)


def _train_grpo_openenv(args: argparse.Namespace, source: str, rows: list[dict[str, Any]]) -> str:
    from transformers import AutoTokenizer
    from trl import GRPOTrainer

    output = args.output_dir / "grpo_openenv"
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    dataset = _dataset_from_rows(stage_payload("grpo_openenv", rows))
    trainer = GRPOTrainer(
        model=source,
        processing_class=tokenizer,
        train_dataset=dataset,
        environment_factory=_environment_factory(args),
        reward_funcs=final_format_reward,
        args=_grpo_config(args, "grpo_openenv"),
    )
    _save(trainer, tokenizer, output)
    return str(output)


def _train_async_grpo(args: argparse.Namespace, source: str, rows: list[dict[str, Any]]) -> str:
    if not args.vllm_server_url:
        raise ValueError("async_grpo requires --vllm-server-url pointing at the same policy checkpoint")

    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

    output = args.output_dir / "async_grpo"
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    dataset = _dataset_from_rows(stage_payload("async_grpo", rows))
    steps = 1 if args.smoke else args.max_steps
    config = AsyncGRPOConfig(
        output_dir=str(output),
        max_steps=steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        vllm_server_base_url=args.vllm_server_url,
        chat_template_kwargs={"enable_thinking": False},
        save_strategy="steps",
        save_steps=max(1, steps),
        report_to="none",
        log_completions=True,
        logging_steps=1,
        seed=args.seed,
    )
    trainer = AsyncGRPOTrainer(
        model=source,
        processing_class=tokenizer,
        train_dataset=dataset,
        environment_factory=_environment_factory(args),
        reward_funcs=final_format_reward,
        args=config,
    )
    _save(trainer, tokenizer, output)
    return str(output)


def ________(self):pass
def train(args: argparse.Namespace) -> None:
    validate_datasets(args.data_dir)##===================================
    stages = selected_stages(args.stages)##===================================
    source = resolve_model_source(args.student_model, args.allow_model_download, "student_model")##===================================
    teacher: str | None = None
    if "distillation" in stages:
        teacher = resolve_model_source(args.teacher_model, args.allow_model_download, "teacher_model")##===================================



    for stage in stages:
        rows = load_jsonl(args.data_dir / STAGE_FILES[stage])
        print(f"\n=== {stage}: {len(rows)} rows, source={source} ===", flush=True)



        if stage == "sft":
            source = _train_sft(args, source, rows)##===================================
        elif stage == "distillation":
            if teacher is None:
                raise AssertionError("teacher model was not resolved")
            source = _train_distillation(args, source, teacher, rows)##===================================
        elif stage == "grpo_tools":
            source = _train_grpo_tools(args, source, rows)##===================================
        elif stage == "grpo_openenv":
            source = _train_grpo_openenv(args, source, rows)##===================================
        elif stage == "async_grpo":
            source = _train_async_grpo(args, source, rows)##===================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="one optimizer step per selected stage")
    parser.add_argument("--allow-model-download", action="store_true")
    ##
    parser.add_argument("--student-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stages", nargs="+", default=["all"], choices=("all", *STAGE_ORDER)
    )
    parser.add_argument("--vllm-server-url", default="")##===================================
    parser.add_argument("--openenv-url", default="",##===================================
        help="use the typed OpenEnv server instead of the in-process environment",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=8)##===================================
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    counts = validate_datasets(args.data_dir)##===================================
    stages = selected_stages(args.stages)##===================================
    source = args.student_model##=======
    plan: list[dict[str, Any]] = []
    for stage in stages:
        output = args.output_dir / stage
        entry: dict[str, Any] = {
            "stage": stage,
            "trainer": TRAINERS[stage],
            "rows": counts[stage],
            "data": str(args.data_dir / STAGE_FILES[stage]),
            "model_input": source,##=======
            "model_output": str(output),
            "multi_turn": True,
        }
        if stage == "distillation":
            entry["teacher_model"] = args.teacher_model
            entry["signal"] = "teacher next-token distribution on student-generated tool trajectories"##===================================
        elif stage == "grpo_tools":
            entry["signal"] = "exact-total and required-tool rewards"
        elif stage in {"grpo_openenv", "async_grpo"}:
            entry["signal"] = "environment-owned binary final-state reward"
            entry["environment"] = (
                f"OpenEnv server at {args.openenv_url}"
                if args.openenv_url
                else "in-process OrderFulfillmentEnv"
            )
        plan.append(entry)
        source = str(output)##=======
    return {
        "mode": "dry-run: no ML imports, checkpoint loads, writes, or network access",
        "stages": plan,
        "async_requires": "Linux, separate vLLM GPU/server, transformers>=5.2.0",
        "download_allowed": bool(args.allow_model_download),
    }
def main() -> None:
    args = parse_args()##========
    if args.dry_run:
        print(json.dumps(build_plan(args), ensure_ascii=False, indent=2))##===================================
        return
    train(args)##===================================
if __name__ == "__main__":
    main()

