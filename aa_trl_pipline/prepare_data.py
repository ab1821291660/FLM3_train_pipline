"""Generate five deterministic 20-row datasets for the Agent RL curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aa_trl_pipline import STAGE_ORDER
from aa_trl_pipline.order_env import replay_oracle
from aa_trl_pipline.toolkit import (
    CATALOG,
    SHIPPING_FEES,
    TOOL_SCHEMAS,
    calculate_grand_total,
    calculate_line_total,
    lookup_product,
    order_totals,
    shipping_quote,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STAGE_FILES = {
    "sft": "sft.jsonl",
    "distillation": "distillation.jsonl",
    "grpo_tools": "grpo_tools.jsonl",
    "grpo_openenv": "grpo_openenv.jsonl",
    "async_grpo": "async_grpo.jsonl",
}
REQUIRED_TOOLS = [
    "lookup_product",
    "calculate_line_total",
    "shipping_quote",
    "calculate_grand_total",
]


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _tool_result(name: str, content: str) -> dict[str, str]:
    return {"role": "tool", "name": name, "content": content}


def task_specs() -> list[dict[str, Any]]:
    """Create 20 unique two-line orders with deterministic expected totals."""

    skus = list(CATALOG)
    cities = list(SHIPPING_FEES)
    rows: list[dict[str, Any]] = []
    for index in range(20):
        sku_a = skus[index % len(skus)]
        sku_b = skus[(index * 3 + 2) % len(skus)]
        if sku_a == sku_b:
            sku_b = skus[(skus.index(sku_b) + 1) % len(skus)]
        qty_a = index % 3 + 1
        qty_b = (index * 2) % 3 + 1
        city = cities[index % len(cities)]
        totals = order_totals(sku_a, qty_a, sku_b, qty_b, city)
        rows.append(
            {
                "index": index,
                "sku_a": sku_a,
                "qty_a": qty_a,
                "sku_b": sku_b,
                "qty_b": qty_b,
                "city": city,
                **totals,
            }
        )
    return rows


def user_instruction(task: dict[str, Any], stateful: bool = False) -> str:
    if stateful:
        return (
            f"请在订单环境中完成采购：{task['sku_a']} × {task['qty_a']}，"
            f"{task['sku_b']} × {task['qty_b']}，收货城市为{task['city']}。"
            "必须先逐个 inspect_product，再逐个 add_to_cart，然后 set_destination，最后 checkout；"
            "根据工具返回结果，用 TOTAL=xx.xx 结束回答。"
        )
    return (
        f"请核算订单：{task['sku_a']} × {task['qty_a']}，{task['sku_b']} × {task['qty_b']}，"
        f"配送到{task['city']}。必须分轮调用工具：先分别 lookup_product，再分别 "
        "calculate_line_total，汇总后调用 shipping_quote，最后调用 calculate_grand_total。"
        "不要心算或跳步，用 TOTAL=xx.xx 结束回答。"
    )


def sft_row(task: dict[str, Any]) -> dict[str, Any]:
    subtotal = float(task["merchandise_total"])
    shipping = float(task["shipping_fee"])
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是订单核算 Agent。以工具结果为准，保留两位小数，并完成多轮工具调用。",
        },
        {"role": "user", "content": user_instruction(task)},
        _tool_call("lookup_product", {"sku": task["sku_a"]}),
        _tool_result("lookup_product", lookup_product(task["sku_a"])),
        _tool_call("lookup_product", {"sku": task["sku_b"]}),
        _tool_result("lookup_product", lookup_product(task["sku_b"])),
        _tool_call(
            "calculate_line_total", {"sku": task["sku_a"], "quantity": task["qty_a"]}
        ),
        _tool_result(
            "calculate_line_total", calculate_line_total(task["sku_a"], task["qty_a"])
        ),
        _tool_call(
            "calculate_line_total", {"sku": task["sku_b"], "quantity": task["qty_b"]}
        ),
        _tool_result(
            "calculate_line_total", calculate_line_total(task["sku_b"], task["qty_b"])
        ),
        _tool_call("shipping_quote", {"city": task["city"], "merchandise_total": subtotal}),
        _tool_result("shipping_quote", shipping_quote(task["city"], subtotal)),
        _tool_call(
            "calculate_grand_total",
            {"merchandise_total": subtotal, "shipping_fee": shipping},
        ),
        _tool_result(
            "calculate_grand_total", calculate_grand_total(subtotal, shipping)
        ),
        {
            "role": "assistant",
            "content": (
                f"两条商品合计 {task['merchandise_total']} CNY，运费 {task['shipping_fee']} CNY。"
                f" TOTAL={task['grand_total']}"
            ),
        },
    ]
    return {
        "id": f"sft-{task['index']:02d}",
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "expected_total": task["grand_total"],
    }


def prompt_row(task: dict[str, Any], stage: str) -> dict[str, Any]:
    prompt = [
        {
            "role": "system",
            "content": "你是可靠的订单 Agent。必须使用给定工具逐步获取事实，不得编造工具结果。",
        },
        {"role": "user", "content": user_instruction(task)},
    ]
    return {
        "id": f"{stage}-{task['index']:02d}",
        "prompt": prompt,
        "expected_total": task["grand_total"],
        "required_tools": REQUIRED_TOOLS,
    }


def environment_row(task: dict[str, Any], stage: str) -> dict[str, Any]:
    return {
        "id": f"{stage}-{task['index']:02d}",
        "prompt": [
            {
                "role": "system",
                "content": "你是订单执行 Agent。环境有状态，必须观察、修改状态并结账。",
            },
            {"role": "user", "content": user_instruction(task, stateful=True)},
        ],
        "sku_a": task["sku_a"],
        "qty_a": task["qty_a"],
        "sku_b": task["sku_b"],
        "qty_b": task["qty_b"],
        "city": task["city"],
        "expected_total": task["grand_total"],
    }


def build_all_rows() -> dict[str, list[dict[str, Any]]]:
    tasks = task_specs()
    return {
        "sft": [sft_row(task) for task in tasks],
        "distillation": [prompt_row(task, "distillation") for task in tasks],
        "grpo_tools": [prompt_row(task, "grpo_tools") for task in tasks],
        "grpo_openenv": [environment_row(task, "grpo_openenv") for task in tasks],
        "async_grpo": [environment_row(task, "async_grpo") for task in tasks],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_datasets(data_dir: Path = DATA_DIR) -> dict[str, int]:
    data_dir.mkdir(parents=True, exist_ok=True)
    all_rows = build_all_rows()
    for stage, rows in all_rows.items():
        path = data_dir / STAGE_FILES[stage]
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return validate_datasets(data_dir)


def ________(self):pass
def _validate_sft(row: dict[str, Any]) -> None:
    messages = row["messages"]
    call_turns = [message for message in messages if message.get("tool_calls")]
    result_turns = [message for message in messages if message.get("role") == "tool"]
    if len(call_turns) < 4 or len(result_turns) < 4:
        raise ValueError(f"{row['id']} is not a multi-turn tool trace")
    if len(row.get("tools", [])) != len(TOOL_SCHEMAS):##===================================##===================================
        raise ValueError(f"{row['id']} has incomplete tool schemas")


def _validate_prompt(row: dict[str, Any]) -> None:
    if not isinstance(row.get("prompt"), list) or len(row["prompt"]) < 2:
        raise ValueError(f"{row['id']} is not a conversational prompt")
    if len(row.get("required_tools", [])) < 2:
        raise ValueError(f"{row['id']} does not require multiple tools")


def validate_datasets(data_dir: Path = DATA_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stage in STAGE_ORDER:
        path = data_dir / STAGE_FILES[stage]
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = load_jsonl(path)##===================================
        if len(rows) != 20:
            raise ValueError(f"{path.name}: expected 20 rows, got {len(rows)}")
        ids = [row.get("id") for row in rows]
        if len(set(ids)) != 20 or None in ids:
            raise ValueError(f"{path.name}: ids must be present and unique")
        # STAGE_ORDER = (
        #     "sft",
        #     "distillation",
        #     "grpo_tools",
        #     "grpo_openenv",
        #     "async_grpo",
        # )
        if stage == "sft":
            for row in rows:
                _validate_sft(row)##===================================
        elif stage in {"distillation",
                       "grpo_tools"}:
            for row in rows:
                _validate_prompt(row)##===================================
        else:
            for row in rows:
                result, reward = replay_oracle(row)##===================================
                if reward != 1.0 or str(row["expected_total"]) not in result:
                    raise ValueError(f"{row['id']}: environment oracle failed")
        counts[stage] = len(rows)
    return counts##===================================##===================================##===================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--check", action="store_true", help="validate existing files without rewriting")
    args = parser.parse_args()
    counts = validate_datasets(args.data_dir) if args.check  else write_datasets(args.data_dir)##===================================
    for stage in STAGE_ORDER:
        print(f"{stage}: {counts[stage]} rows -> {args.data_dir / STAGE_FILES[stage]}")
if __name__ == "__main__":
    main()

