import json
import base64
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from fault_injection.sglang_weight_bitflip import (
    LOADER_PATH,
    build_update_payload,
    decode_control_name,
    encode_control_name,
    sdc_weight_fault_loader,
)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.model.proj.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
            )


def control(command):
    return [(encode_control_name(command), torch.zeros(1, dtype=torch.uint8))]


class SGLangWeightBitflipTest(unittest.TestCase):
    def test_control_round_trip(self):
        command = {"action": "inject", "module_name": "模型.layer", "bits": [1, 14]}
        self.assertEqual(decode_control_name(encode_control_name(command)), command)

    def test_update_payload_is_by_value_and_target_only(self):
        command = {"action": "restore", "fault_id": "f", "event_path": "/tmp/e"}
        payload = build_update_payload(command)
        raw = base64.b64decode(payload["serialized_named_tensors"][0])
        items = pickle.loads(raw)
        self.assertEqual(decode_control_name(items[0][0]), command)
        self.assertIsInstance(items[0][1], torch.Tensor)
        self.assertEqual(payload["load_format"], LOADER_PATH)
        self.assertTrue(payload["disable_draft_model"])
        self.assertTrue(payload["abort_all_requests"])
        self.assertTrue(payload["flush_cache"])

    def test_inject_and_restore_exact_bits(self):
        with tempfile.TemporaryDirectory() as root:
            model = ToyModel()
            inject_event = Path(root) / "inject.json"
            restore_event = Path(root) / "restore.json"
            weight = model.model.proj.weight
            before = weight.clone()
            before_bits = int(weight.view(torch.int16).reshape(-1)[1].item()) & 0xFFFF

            sdc_weight_fault_loader(
                model,
                control(
                    {
                        "action": "inject",
                        "fault_id": "unit_weight",
                        "module_name": "model.proj",
                        "parameter_name": "weight",
                        "flat_index": 1,
                        "bit_positions": [1, 14],
                        "event_path": str(inject_event),
                    }
                ),
            )

            after_bits = int(weight.view(torch.int16).reshape(-1)[1].item()) & 0xFFFF
            self.assertEqual(after_bits, before_bits ^ ((1 << 1) | (1 << 14)))
            event = json.loads(inject_event.read_text(encoding="utf-8"))
            self.assertEqual(event["status"], "injected")
            self.assertEqual(event["before_bits_hex"], f"0x{before_bits:04x}")
            self.assertEqual(event["after_bits_hex"], f"0x{after_bits:04x}")

            sdc_weight_fault_loader(
                model,
                control(
                    {
                        "action": "restore",
                        "fault_id": "unit_weight",
                        "event_path": str(restore_event),
                    }
                ),
            )

            self.assertTrue(torch.equal(weight, before))
            restored = json.loads(restore_event.read_text(encoding="utf-8"))
            self.assertEqual(restored["status"], "restored")
            self.assertEqual(restored["restored_bits_hex"], f"0x{before_bits:04x}")

    def test_rejects_second_fault_until_restore(self):
        with tempfile.TemporaryDirectory() as root:
            model = ToyModel()
            command = {
                "action": "inject",
                "fault_id": "first",
                "module_name": "model.proj",
                "flat_index": 0,
                "bit_positions": [0],
                "event_path": str(Path(root) / "first.json"),
            }
            sdc_weight_fault_loader(model, control(command))
            command["fault_id"] = "second"
            command["event_path"] = str(Path(root) / "second.json")
            with self.assertRaises(RuntimeError):
                sdc_weight_fault_loader(model, control(command))

    def test_restore_refuses_unexpected_intermediate_change(self):
        with tempfile.TemporaryDirectory() as root:
            model = ToyModel()
            sdc_weight_fault_loader(
                model,
                control(
                    {
                        "action": "inject",
                        "fault_id": "guarded",
                        "module_name": "model.proj",
                        "flat_index": 0,
                        "bit_positions": [0],
                        "event_path": str(Path(root) / "inject.json"),
                    }
                ),
            )
            bits = model.model.proj.weight.view(torch.int16).reshape(-1)
            with torch.no_grad():
                bits[0] = bits[0] ^ 2
            with self.assertRaises(RuntimeError):
                sdc_weight_fault_loader(
                    model,
                    control(
                        {
                            "action": "restore",
                            "fault_id": "guarded",
                            "event_path": str(Path(root) / "restore.json"),
                        }
                    ),
                )

    def test_rejects_unknown_module(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(KeyError):
                sdc_weight_fault_loader(
                    ToyModel(),
                    control(
                        {
                            "action": "inject",
                            "fault_id": "missing",
                            "module_name": "model.missing",
                            "flat_index": 0,
                            "bit_positions": [0],
                            "event_path": str(Path(root) / "missing.json"),
                        }
                    ),
                )


if __name__ == "__main__":
    unittest.main()
