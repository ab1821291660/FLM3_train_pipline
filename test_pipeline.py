from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from lfm3_tiny import ByteTokenizer, TinyLFM3Config, TinyLFM3ForCausalLM
from prepare_data import EXPECTED_FILES, check_datasets
from train_lora import model_source
from train_pipeline import ROOT, STAGE_ORDER, validate_stage_schemas


class DatasetTests(unittest.TestCase):
    def test_every_stage_has_twenty_valid_rows(self) -> None:
        counts = validate_stage_schemas(ROOT / "data")
        self.assertEqual(set(counts), set(STAGE_ORDER))
        self.assertTrue(all(count == 20 for count in counts.values()))
        self.assertEqual(len(EXPECTED_FILES), 6)


class ArchitectureTests(unittest.TestCase):
    def test_tri_hybrid_forward_and_backward(self) -> None:
        config = TinyLFM3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            layer_types=["conv", "sliding_attention", "full_attention"],
        )
        model = TinyLFM3ForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 24))
        output = model(input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)
        self.assertEqual(tuple(output["logits"].shape), (2, 24, config.vocab_size))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        self.assertEqual(model.architecture_summary()["conv"], 1)
        self.assertEqual(model.architecture_summary()["sliding_attention"], 1)
        self.assertEqual(model.architecture_summary()["full_attention"], 1)

    def test_byte_tokenizer_round_trip(self) -> None:
        tokenizer = ByteTokenizer()
        text = "LFM3 端侧"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


class DownloadGuardTests(unittest.TestCase):
    def test_config_only_directory_is_not_treated_as_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text("{}", encoding="utf-8")
            args = Namespace(local_model_dir=root, allow_weight_download=False, model_id="LiquidAI/LFM3-1.7B")
            with self.assertRaises(FileNotFoundError):
                model_source(args)

    def test_remote_checkpoint_requires_explicit_permission(self) -> None:
        args = Namespace(local_model_dir=None, allow_weight_download=False, model_id="LiquidAI/LFM3-1.7B")
        with self.assertRaises(RuntimeError):
            model_source(args)


if __name__ == "__main__":
    check_datasets(ROOT / "data")
    unittest.main()
