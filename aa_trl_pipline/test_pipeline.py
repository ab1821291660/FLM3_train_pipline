from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from aa_trl_pipline import STAGE_ORDER
from aa_trl_pipline.order_env import OrderFulfillmentEnv, replay_oracle
from aa_trl_pipline.prepare_data import DATA_DIR, STAGE_FILES, load_jsonl, validate_datasets
from aa_trl_pipline.rewards import exact_total_reward, required_tools_reward
from aa_trl_pipline.train_pipeline import build_plan, resolve_model_source


class DatasetTests(unittest.TestCase):
    def test_every_stage_has_twenty_rows(self) -> None:
        counts = validate_datasets(DATA_DIR)
        self.assertEqual(tuple(counts), STAGE_ORDER)
        self.assertTrue(all(count == 20 for count in counts.values()))

    def test_sft_rows_are_multi_turn_tool_trajectories(self) -> None:
        rows = load_jsonl(DATA_DIR / STAGE_FILES["sft"])
        for row in rows:
            call_turns = [message for message in row["messages"] if message.get("tool_calls")]
            tool_turns = [message for message in row["messages"] if message.get("role") == "tool"]
            self.assertGreaterEqual(len(call_turns), 6)
            self.assertEqual(len(call_turns), len(tool_turns))
            self.assertIn(f"TOTAL={row['expected_total']}", row["messages"][-1]["content"])

    def test_prompt_stages_require_multiple_tools(self) -> None:
        for stage in ("distillation", "grpo_tools"):
            rows = load_jsonl(DATA_DIR / STAGE_FILES[stage])
            for row in rows:
                self.assertGreaterEqual(len(row["required_tools"]), 4)
                self.assertEqual(row["prompt"][-1]["role"], "user")


class EnvironmentTests(unittest.TestCase):
    def test_all_environment_oracles_finish_with_reward_one(self) -> None:
        for stage in ("grpo_openenv", "async_grpo"):
            for row in load_jsonl(DATA_DIR / STAGE_FILES[stage]):
                result, reward = replay_oracle(row)
                self.assertEqual(reward, 1.0)
                self.assertIn(str(row["expected_total"]), result)

    def test_wrong_cart_has_zero_reward(self) -> None:
        row = load_jsonl(DATA_DIR / STAGE_FILES["grpo_openenv"])[0]
        env = OrderFulfillmentEnv()
        env.reset(**{key: row[key] for key in ("sku_a", "qty_a", "sku_b", "qty_b", "city", "expected_total")})
        env.inspect_product(row["sku_a"])
        env.add_to_cart(row["sku_a"], row["qty_a"] + 1)
        env.inspect_product(row["sku_b"])
        env.add_to_cart(row["sku_b"], row["qty_b"])
        env.set_destination(row["city"])
        env.checkout()
        self.assertEqual(env.get_reward(), 0.0)


class RewardTests(unittest.TestCase):
    def test_structured_completion_rewards(self) -> None:
        completion = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "lookup_product", "arguments": {}}},
                    {"type": "function", "function": {"name": "shipping_quote", "arguments": {}}},
                ],
            },
            {"role": "assistant", "content": "核算完成。 TOTAL=42.50"},
        ]
        self.assertEqual(exact_total_reward([completion], ["42.50"]), [1.0])
        self.assertEqual(
            required_tools_reward([completion], [["lookup_product", "shipping_quote"]]), [1.0]
        )


class GuardAndPlanTests(unittest.TestCase):
    def test_remote_model_is_download_gated(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_model_source("Qwen/Qwen3-0.6B", False, "student_model")

    def test_config_only_directory_is_not_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                resolve_model_source(str(root), False, "student_model")

    def test_dry_plan_has_five_linked_stages(self) -> None:
        args = Namespace(
            data_dir=DATA_DIR,
            stages=["all"],
            output_dir=DATA_DIR.parent / "outputs",
            student_model="Qwen/Qwen3-0.6B",
            teacher_model="Qwen/Qwen3-1.7B",
            allow_model_download=False,
            openenv_url="",
        )
        plan = build_plan(args)
        self.assertEqual([item["stage"] for item in plan["stages"]], list(STAGE_ORDER))
        for previous, current in zip(plan["stages"], plan["stages"][1:]):
            self.assertEqual(previous["model_output"], current["model_input"])


if __name__ == "__main__":
    unittest.main()
