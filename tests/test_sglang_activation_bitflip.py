import json
import tempfile
import unittest
from pathlib import Path

import torch

from fault_injection.sglang_activation_bitflip import (
    create_activation_bitflip_hook,
)


class ActivationBitflipHookTest(unittest.TestCase):
    def make_paths(self, root: str) -> tuple[Path, Path]:
        directory = Path(root)
        return directory / "fault.arm", directory / "event.json"

    def test_unarmed_hook_does_not_modify_output(self):
        with tempfile.TemporaryDirectory() as root:
            arm_path, event_path = self.make_paths(root)
            hook = create_activation_bitflip_hook(
                {
                    "arm_path": str(arm_path),
                    "event_path": str(event_path),
                    "bit_positions": [0],
                }
            )
            output = torch.tensor([[1.0]], dtype=torch.bfloat16)
            before = output.clone()

            returned = hook(torch.nn.Identity(), (), output)

            self.assertIs(returned, output)
            self.assertTrue(torch.equal(output, before))
            self.assertFalse(event_path.exists())

    def test_single_bit_flip_records_exact_evidence_and_fires_once(self):
        with tempfile.TemporaryDirectory() as root:
            arm_path, event_path = self.make_paths(root)
            arm_path.touch()
            hook = create_activation_bitflip_hook(
                {
                    "arm_path": str(arm_path),
                    "event_path": str(event_path),
                    "bit_positions": [0],
                    "flat_index": 1,
                    "fault_id": "unit_single",
                    "module_name": "model.layers.20.mlp.down_proj",
                }
            )
            output = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
            before_bits = int(output.view(torch.int16).reshape(-1)[1].item()) & 0xFFFF

            hook(torch.nn.Identity(), (), output)
            after_bits = int(output.view(torch.int16).reshape(-1)[1].item()) & 0xFFFF

            self.assertEqual(after_bits, before_bits ^ 0x0001)
            self.assertFalse(arm_path.exists())
            event = json.loads(event_path.read_text(encoding="utf-8"))
            self.assertEqual(event["status"], "injected")
            self.assertEqual(event["fault_mode"], "single")
            self.assertEqual(event["before_bits_hex"], f"0x{before_bits:04x}")
            self.assertEqual(event["after_bits_hex"], f"0x{after_bits:04x}")
            self.assertEqual(event["call_index"], 1)

            snapshot = output.clone()
            hook(torch.nn.Identity(), (), output)
            self.assertTrue(torch.equal(output, snapshot))

    def test_double_bit_flip_supports_tuple_output_and_negative_index(self):
        with tempfile.TemporaryDirectory() as root:
            arm_path, event_path = self.make_paths(root)
            arm_path.touch()
            hook = create_activation_bitflip_hook(
                {
                    "arm_path": str(arm_path),
                    "event_path": str(event_path),
                    "bit_positions": [1, 14],
                    "flat_index": -1,
                }
            )
            tensor = torch.tensor([[3.0, -4.0]], dtype=torch.bfloat16)
            output = (tensor, None)
            before_bits = int(tensor.view(torch.int16).reshape(-1)[-1].item()) & 0xFFFF

            returned = hook(torch.nn.Identity(), (), output)
            after_bits = int(tensor.view(torch.int16).reshape(-1)[-1].item()) & 0xFFFF

            self.assertIs(returned, output)
            self.assertEqual(after_bits, before_bits ^ ((1 << 1) | (1 << 14)))
            event = json.loads(event_path.read_text(encoding="utf-8"))
            self.assertEqual(event["fault_mode"], "double")
            self.assertEqual(event["output_location"], "tuple[0]")
            self.assertEqual(event["flat_index_normalized"], 1)

    def test_rejects_invalid_bit_positions(self):
        with self.assertRaises(ValueError):
            create_activation_bitflip_hook(
                {
                    "arm_path": "/tmp/a",
                    "event_path": "/tmp/e",
                    "bit_positions": [16],
                }
            )

    def test_rejects_noncontiguous_output(self):
        with tempfile.TemporaryDirectory() as root:
            arm_path, event_path = self.make_paths(root)
            arm_path.touch()
            hook = create_activation_bitflip_hook(
                {
                    "arm_path": str(arm_path),
                    "event_path": str(event_path),
                    "bit_positions": [0],
                }
            )
            output = torch.ones((2, 3), dtype=torch.bfloat16).transpose(0, 1)

            with self.assertRaises(ValueError):
                hook(torch.nn.Identity(), (), output)


if __name__ == "__main__":
    unittest.main()
