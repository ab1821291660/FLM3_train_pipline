"""Run the six-stage, no-weight-download LFM3 training reproduction.

The default model is a tiny random initialization that retains the public
tri-hybrid backbone ideas.  It performs real forward/backward/optimizer steps
on all six local datasets and never contacts Hugging Face.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lfm3_tiny import ByteTokenizer, TinyLFM3Config, TinyLFM3ForCausalLM, format_chat

from prepare_data import EXPECTED_FILES, check_datasets##===================================
ROOT = Path(__file__).resolve().parent
STAGE_ORDER = ("pretrain", "midtrain", "sft", "mixed_rl", "mopd", "agentic_rl")
STAGE_FILES = dict(zip(STAGE_ORDER, EXPECTED_FILES, strict=True))
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(requested)



def ____辅助函数(self):pass
def _assistant_pair(messages: list[dict[str, Any]]) -> tuple[str, str]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("assistant-supervised rows must end with an assistant message")
    prompt = format_chat(messages[:-1]) + "<|im_start|>assistant\n"##===================================
    completion = str(messages[-1].get("content", "")) + "<|im_end|>\n"
    return prompt, completion


def _prompt_completion_ids(
    tokenizer: ByteTokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_eos=False, max_length=max_length)
    available = max_length - len(prompt_ids)
    if available < 2:
        # Preserve a useful completion target when a byte-level prompt is long.
        keep_prompt = max(2, max_length // 2)
        prompt_ids = prompt_ids[:keep_prompt]
        available = max_length - len(prompt_ids)
    completion_ids = tokenizer.encode(completion, add_bos=False, max_length=available)
    ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids)    + completion_ids
    return ids, labels



def ____sft类(self):pass
def _supervised_examples(
    stage: str,
    rows: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    max_length: int,
) -> list[tuple[list[int], list[int]]]:
    examples: list[tuple[list[int], list[int]]] = []
    for row in rows:
        if stage == "pretrain":
            ids = tokenizer.encode(row["text"], max_length=max_length)
            examples.append((ids, ids.copy()))
        elif stage == "midtrain":
            text = format_chat(row["messages"])##===================================
            ids = tokenizer.encode(text, max_length=max_length)
            examples.append((ids, ids.copy()))


        elif stage == "sft":
            prompt, completion = _assistant_pair(row["messages"])##===================================
            examples.append(_prompt_completion_ids(tokenizer, prompt, completion, max_length))##===================================


        elif stage == "mopd":##????明显不对--这个是蒸馏
            prompt = format_chat([{"role": "user", "content": row["prompt"]}]) + "<|im_start|>assistant\n"
            completion = row["teacher_response"] + "<|im_end|>\n"##===================================
            examples.append(_prompt_completion_ids(tokenizer, prompt, completion, max_length))##===================================
        else:
            raise ValueError(f"{stage} is not a supervised stage")
    return examples

def _next_batch(
    examples: list[tuple[list[int], list[int]]],
    step: int,
    batch_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    start = (step * batch_size) % len(examples)
    indices = [(start + offset) % len(examples) for offset in range(batch_size)]
    return [examples[index][0] for index in indices], [examples[index][1] for index in indices]

def train_supervised_stage(
    model: TinyLFM3ForCausalLM,
    tokenizer: ByteTokenizer,
    stage: str,
    rows: list[dict[str, Any]],
    stage_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int | str]:
    max_length = int(training_config["max_seq_length"])
    examples = _supervised_examples(stage, rows, tokenizer, max_length)##===================================


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    losses: list[float] = []
    model.train()
    max_steps = int(stage_config["max_steps"])
    for step in range(max_steps):##10
        sequences, labels = _next_batch(examples, step, int(training_config["batch_size"]))##===================================
        input_ids, attention_mask, label_tensor = tokenizer.batch(sequences, labels, device=device)


        output = model(input_ids, attention_mask=attention_mask, labels=label_tensor)
        loss = output["loss"]
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in {stage}: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "objective": stage_config["objective"],
        "steps": max_steps,
        "examples": len(examples),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
    }




def ____rl类(self):pass
def train_group_relative_stage(
    model: TinyLFM3ForCausalLM,
    tokenizer: ByteTokenizer,
    stage: str,
    rows: list[dict[str, Any]],
    stage_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int | str]:
    reference = copy.deepcopy(model).to(device).eval()##===================================
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )


    beta = float(stage_config.get("beta", 0.0))
    losses: list[float] = []
    reward_means: list[float] = []
    max_steps = int(stage_config["max_steps"])
    max_length = int(training_config["max_seq_length"])
    model.train()
    for step in range(max_steps):
        row = rows[step % len(rows)]
        prompt = format_chat([{"role": "user", "content": row["prompt"]}]) + "<|im_start|>assistant\n"##===================================
        candidates = row["candidates"]##===================================
        completions = [candidate["text"] + "<|im_end|>\n" for candidate in candidates]



        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        rewards = torch.tensor([float(candidate["reward"]) for candidate in candidates], device=device)
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)



        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        log_probs = completion_log_probabilities(model, tokenizer, prompt, completions, max_length, device)##===================================##===================================
        with torch.no_grad():
            reference_log_probs = completion_log_probabilities(##===================================##===================================
                reference, tokenizer, prompt, completions, max_length, device
            )
        policy_loss = -(advantages.detach() * log_probs).mean()##===================================
        # A stable sequence-level KL proxy keeps this tiny offline exercise near
        # the pre-stage policy.  The full-size route uses TRL's native KL term.
        kl_proxy = (log_probs - reference_log_probs).pow(2).mean()##===================================
        ##
        loss = policy_loss + beta * kl_proxy##===================================
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in {stage}: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        reward_means.append(float(rewards.mean().cpu()))
    return {
        "objective": stage_config["objective"],##objective
        "steps": max_steps,
        "examples": len(rows),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "mean_static_reward": sum(reward_means) / len(reward_means),
        "beta": beta,
    }
def completion_log_probabilities(
    model: TinyLFM3ForCausalLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    completions: list[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    examples = [_prompt_completion_ids(tokenizer, prompt, completion, max_length) for completion in completions]
    sequences = [example[0] for example in examples]
    labels = [example[1] for example in examples]
    input_ids, attention_mask, label_tensor = tokenizer.batch(sequences, labels, device=device)
    assert label_tensor is not None


    logits = model(input_ids, attention_mask=attention_mask)["logits"]
    shifted_logits = logits[:, :-1]
    shifted_labels = label_tensor[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    ##
    token_log_probs = F.log_softmax(shifted_logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    lengths = valid.sum(dim=-1).clamp_min(1)
    return (token_log_probs * valid).sum(dim=-1) / lengths##===================================##===================================##===================================




def ________(self):pass
def validate_stage_schemas(data_dir: Path) -> dict[str, int]:
    check_datasets(data_dir)##===================================
    required = {
        "pretrain": {"id", "text", "domain"},
        "midtrain": {"id", "messages", "tools"},
        "sft": {"id", "messages", "category"},
        "mixed_rl": {"id", "prompt", "candidates"},##===================================
        "mopd": {"id", "prompt", "teacher_response",   "teacher_domain"},
        "agentic_rl": {"id", "prompt", "candidates",    "expected_tool"},##===================================
    }
    counts = {}
    for stage in STAGE_ORDER:
        rows = load_jsonl(data_dir / STAGE_FILES[stage])##===================================
        for row in rows:
            missing = required[stage] - row.keys()
            if missing:
                raise ValueError(f"{stage}/{row.get('id')} missing fields: {sorted(missing)}")
        counts[stage] = len(rows)
    return counts
def save_checkpoint(
    model: TinyLFM3ForCausalLM,
    output_dir: Path,
    stage: str,
    metrics: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stage}.pt"
    torch.save(
        {"stage": stage, "config": model.config.to_dict(), "model_state_dict": model.state_dict(), "metrics": metrics},
        path,
    )
    return path
def run(args: argparse.Namespace) -> dict[str, Any]:
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    counts = validate_stage_schemas(args.data_dir)##===================================
    if args.validate_data:
        return {"validated": counts}




    training_config = config["training"]
    seed = int(args.seed if args.seed is not None else training_config["seed"])
    set_seed(seed)##========
    device = resolve_device(args.device)##========
    model_config = TinyLFM3Config(**config["model"])##===================================
    if args.smoke:
        # Keep all three operator types but shrink the CPU validation model.保留所有三种运算符类型，但缩小CPU验证模型。
        model_config.hidden_size = 32
        model_config.intermediate_size = 64
        model_config.num_hidden_layers = 3
        model_config.num_attention_heads = 4
        model_config.num_key_value_heads = 2
        model_config.layer_types = ["conv", "sliding_attention", "full_attention"]
        model_config.max_position_embeddings = min(model_config.max_position_embeddings, 192)
        training_config["max_seq_length"] = min(int(training_config["max_seq_length"]), 192)
        for stage_config in training_config["stages"].values():
            stage_config["max_steps"] = 1
    model = TinyLFM3ForCausalLM(model_config).to(device)##===================================




    tokenizer = ByteTokenizer()##vocab_size = 259
    selected = STAGE_ORDER if args.stages == ["all"] else tuple(args.stages)
    unknown = set(selected) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    run_metrics: dict[str, Any] = {
        "seed": seed,
        "device": str(device),
        "architecture": model.architecture_summary(),
        "data_counts": counts,
        "stages": {},
        "reproduction_boundary": (
            "Tiny random-initialized architecture and offline objective surrogates; "
            "not Liquid AI's undisclosed datasets, teachers, environments, or compute."
        ),
    }
    print("architecture:", json.dumps(run_metrics["architecture"], ensure_ascii=False, sort_keys=True))
    started = time.perf_counter()
    for stage in selected:
        rows = load_jsonl(args.data_dir / STAGE_FILES[stage])##===================================


        stage_config = training_config["stages"][stage]
        stage_started = time.perf_counter()
        if stage in {"mixed_rl", "agentic_rl"}:
            metrics = train_group_relative_stage(##===================================##===================================
                model, tokenizer, stage, rows, stage_config, training_config, device
            )
        else:
            metrics = train_supervised_stage(##===================================##===================================
                model, tokenizer, stage, rows, stage_config, training_config, device
            )
        metrics["seconds"] = time.perf_counter() - stage_started
        run_metrics["stages"][stage] = metrics
        print(f"[{stage:10s}] " + json.dumps(metrics, ensure_ascii=False, sort_keys=True))
        if not args.no_save:
            checkpoint = save_checkpoint(model, args.output_dir, stage, metrics)##===================================
            print(f"checkpoint  {checkpoint}")
    run_metrics["seconds"] = time.perf_counter() - started
    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(run_metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"metrics     {metrics_path}")
    return run_metrics
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline_config.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tiny-pipeline")
    parser.add_argument("--stages", nargs="+", default=["all"], help="all or any ordered subset of stages")
    parser.add_argument("--seed", type=int)
    ##
    parser.add_argument("--smoke", action="store_true", help="one optimizer step per stage with a smaller model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-save", action="store_true", help="run without writing checkpoints or metrics")
    parser.add_argument("--validate-data", action="store_true", help="validate all datasets and exit")
    return parser.parse_args()
def main() -> None:
    args = parse_args()##===================================
    metrics = run(args)##===================================
    if args.validate_data:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
# python prepare_data.py --check
# python train_pipeline.py --smoke --device cpu --no-save
# python -m unittest -v test_pipeline.py

