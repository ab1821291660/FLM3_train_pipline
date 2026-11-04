"""Optional LFM3-1.7B LoRA route based on Liquid AI's public TRL notebooks.

This entry point is deliberately download-gated.  Without an existing local
checkpoint, it refuses to continue unless --allow-weight-download is passed.
The repository validation never passes that flag and therefore never downloads
the 1.7B weights.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from prepare_data import check_datasets
from train_pipeline import ROOT, STAGE_FILES, STAGE_ORDER, load_jsonl


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]


def has_checkpoint_weights(path: Path) -> bool:
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    return path.is_dir() and any(next(path.glob(pattern), None) is not None for pattern in patterns)
def model_source(args: argparse.Namespace) -> str:
    if args.local_model_dir is not None:
        local = args.local_model_dir.resolve()
        if not has_checkpoint_weights(local):##========
            raise FileNotFoundError(
                f"{local} has config/tokenizer files but no checkpoint weights; "
                "use --dry-run or provide a real local checkpoint"
            )
        return str(local)
    if not args.allow_weight_download:
        raise RuntimeError(
            "Refusing to download LFM3 weights. Provide --local-model-dir, "
            "or explicitly pass --allow-weight-download on a machine intended for full LoRA training."
        )
    return args.model_id


def ____辅助函数(self):pass
def prepare_sft_rows(stage: str, rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    prepared = []
    for row in rows:
        if stage == "pretrain":##========
            text = row["text"]
        elif stage in {"midtrain",##========
                       "sft"}:##========
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)

        elif stage == "mopd":##========
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["teacher_response"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            raise ValueError(stage)
        prepared.append({"text": text})
    return prepared


def prepare_dpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        ranked = sorted(row["candidates"], key=lambda item: float(item["reward"]))
        prepared.append(
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],##===================================
                "chosen": [{"role": "assistant", "content": ranked[-1]["text"]}],##===================================##===================================
                "rejected": [{"role": "assistant", "content": ranked[0]["text"]}],##===================================##===================================
            }
        )
    return prepared


def prepare_grpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "expected_tool": row["expected_tool"],
        }
        for row in rows
    ]




def ________(self):pass
def train(args: argparse.Namespace) -> None:
    source = model_source(args)##===================================
    # Heavy imports are below the explicit checkpoint/download guard.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer, \
        GRPOConfig, GRPOTrainer,\
        SFTConfig, SFTTrainer

    device_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=not args.allow_weight_download)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        trust_remote_code=True,
        dtype=device_dtype,
        device_map="auto",
        local_files_only=not args.allow_weight_download,
    )




    selected = STAGE_ORDER if args.stages == ["all"]  else tuple(args.stages)
    for stage in selected:
        rows = load_jsonl(args.data_dir / STAGE_FILES[stage])
        peft_config = None
        if not isinstance(model, PeftModel):
            rank = 64 if stage == "pretrain" else 32 if stage == "agentic_rl" else 16
            peft_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.0 if stage in {"pretrain", "agentic_rl"} else 0.05,
                target_modules=LORA_TARGETS,
                task_type="CAUSAL_LM",
            )
        stage_output = args.output_dir / stage
        if stage in {"pretrain", "midtrain", "sft", "mopd"}:
            dataset = Dataset.from_list(prepare_sft_rows(stage, rows, tokenizer))##===================================



            trainer = SFTTrainer(
                model=model,
                args=SFTConfig(
                    output_dir=str(stage_output),
                    num_train_epochs=args.epochs,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    learning_rate=2e-4,
                    max_length=args.max_length,
                    dataset_text_field="text",
                    packing=stage in {"pretrain", "midtrain"},##====
                    save_strategy="no",
                    report_to="none",
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
            )
        elif stage == "mixed_rl":
            dataset = Dataset.from_list(prepare_dpo_rows(rows))##===================================
            ##
            trainer = DPOTrainer(
                model=model,
                args=DPOConfig(
                    output_dir=str(stage_output),
                    num_train_epochs=args.epochs,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=8,
                    learning_rate=2e-5,
                    beta=0.1,
                    max_length=args.max_length,
                    save_strategy="no",
                    report_to="none",
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
            )
        else:
            dataset = Dataset.from_list(prepare_grpo_rows(rows))##===================================
            ##
            def tool_reward(prompts, completion_ids, expected_tool, **kwargs):
                del prompts, kwargs
                texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
                return [1.0 if expected in text else -1.0 for expected, text in zip(expected_tool, texts)]
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=tool_reward,##===================================
                args=GRPOConfig(
                    output_dir=str(stage_output),
                    max_steps=args.grpo_steps,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=16,
                    num_generations=4,
                    max_completion_length=256,
                    learning_rate=1e-5,
                    beta=0.04,
                    save_strategy="no",
                    report_to="none",
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
            )
        trainer.train()
        trainer.save_model(str(stage_output))  # saves the LoRA adapter, not a merged 1.7B checkpoint
        model = trainer.model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="LiquidAI/LFM3-1.7B-Base")
    parser.add_argument("--local-model-dir", type=Path)
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan without model imports")
    ##
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "lora-pipeline")
    parser.add_argument("--stages", nargs="+", default=["all"], choices=("all", *STAGE_ORDER))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--grpo-steps", type=int, default=20)
    return parser.parse_args()
def print_plan(args: argparse.Namespace) -> None:
    check_datasets(args.data_dir)
    selected = STAGE_ORDER if args.stages == ["all"] else tuple(args.stages)
    plan = {
        "mode": "dry-run; no model imports and no network access",
        "model_id": args.model_id,
        "local_model_dir": None if args.local_model_dir is None  else str(args.local_model_dir),
        "stages": [
            {
                "name": stage,
                "rows": len(load_jsonl(args.data_dir / STAGE_FILES[stage])),
                "trainer": (
                    "DPOTrainer" if stage == "mixed_rl" else "GRPOTrainer" if stage == "agentic_rl" else "SFTTrainer"
                    # if stage == "mixed_rl":
                    #     trainer_name = "DPOTrainer"##===================================
                    # elif stage == "agentic_rl":
                    #     trainer_name = "GRPOTrainer"##===================================
                    # else:
                    #     trainer_name = "SFTTrainer"##===================================
                ),
            }
            for stage in selected
        ],
        "download_allowed": bool(args.allow_weight_download),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
def main() -> None:
    args = parse_args()
    if args.dry_run:
        print_plan(args)##===================================
    else:
        check_datasets(args.data_dir)##========
        train(args)##===================================
if __name__ == "__main__":
    main()


