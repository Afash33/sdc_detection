"""CPU isolation/lifecycle checks; real SGLang GPU integration remains untested."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

fault = importlib.import_module("fault_injection.2026-10-09_sglang_draft_bitflip")
client = importlib.import_module("experiments.2026-10-06_sglang_baseline")


class Mode:
    def __init__(self, name="EXTEND"):
        self.name = name

    def is_idle(self):
        return self.name == "IDLE"


class TupleLinear(nn.Linear):
    def forward(self, value):
        return super().forward(value), None


class Draft(nn.Module):
    def __init__(self, layers=1):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=4)
        self.fc = nn.Linear(12, 4, bias=False, dtype=torch.bfloat16)
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.down_proj = TupleLinear(4, 4, bias=False, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([layer])
        with torch.no_grad():
            self.fc.weight.fill_(0.0625)
            layer.mlp.down_proj.weight.copy_(torch.eye(4, dtype=torch.bfloat16))

    def forward(self, tokens, positions, batch):
        value = batch.spec_info.hidden_states
        if value.shape[-1] != 4:
            value = self.fc(value)
        return self.layers[0].mlp.down_proj(value)


class DraftFaultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"context_path": str(self.root / "2026-10-09_active_sample.json"),
                       "arm_path": str(self.root / "2026-10-09_draft_fault_arm.json")}
        self.tokens = torch.tensor([10, 20])
        self.positions = torch.tensor([0, 1])
        self.features = torch.full((2, 12), 0.5, dtype=torch.bfloat16)
        self.batch = SimpleNamespace(forward_mode=Mode(),
                                     spec_info=SimpleNamespace(hidden_states=self.features))
        self.model = Draft()
        self.model.register_forward_hook(fault.create_draft_fault_hook(self.config))
        self.model(self.tokens, self.positions, self.batch)  # Unarmed warmup.

    def tearDown(self):
        self.temp.cleanup()

    def arm(self, point, *, rid="request", phase="prefill", occurrence=1, bit=7):
        trace = self.root / f"2026-10-09_trace_{rid}.jsonl"
        client.write_json(Path(self.config["context_path"]),
                          {"rid": rid, "sample_id": 0, "trace_path": str(trace)})
        command = {"fault_id": rid + point, "rid": rid, "point": point, "phase": phase,
                   "event_path": str(self.root / f"2026-10-09_event_{rid}.json"),
                   "bit_positions": [bit], "tap": "high", "row": -1,
                   "channel": 0, "occurrence": occurrence}
        client.write_json(Path(self.config["arm_path"]), command)
        return command, trace

    def test_interface_clones_aliases_and_targets_correct_tap(self):
        alias = self.features[:, :]
        original = self.features.view(torch.int16).clone()
        for tap, offset in (("low", 0), ("middle", 4), ("high", 8)):
            command = {"point": "interface", "tap": tap, "bit_positions": [7],
                       "row": -1, "channel": 1}
            changed, evidence = fault.copy_and_flip(self.features, command, 4)
            changed_indices = torch.nonzero(changed.view(torch.int16) != original).tolist()
            self.assertEqual(changed_indices, [[1, offset + 1]])
            self.assertEqual(int(original[1, offset + 1]) ^ 128,
                             int(changed.view(torch.int16)[1, offset + 1]))
            self.assertNotEqual(changed.data_ptr(), self.features.data_ptr())
            self.assertTrue(torch.equal(alias.view(torch.int16), original))
            self.assertEqual(evidence["original_sha256_before"], evidence["original_sha256_after"])

    def test_nonfinite_fault_evidence_is_strict_json(self):
        tensor = torch.ones(1, 4, dtype=torch.bfloat16)
        changed, evidence = fault.copy_and_flip(tensor,
                    {"point": "fusion", "bit_positions": [14], "row": 0}, 4)
        self.assertTrue(torch.isinf(changed[0, 0]))
        self.assertEqual(evidence["after_value_class"], "positive_inf")
        self.assertIsNone(evidence["after_value"])
        json.dumps(evidence, allow_nan=False)
        self.assertTrue(torch.equal(tensor, torch.ones_like(tensor)))

    def test_each_point_fires_once_preserving_parameters_and_target_features(self):
        parameters = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        original = self.features.view(torch.int16).clone()
        expected = self.model(self.tokens, self.positions, self.batch)[0]
        for point in fault.POINTS:
            command, trace = self.arm(point, rid=point)
            output, bias = self.model(self.tokens, self.positions, self.batch)
            self.assertIsNone(bias)
            self.assertFalse(torch.equal(output, expected))
            self.assertTrue(torch.equal(self.features.view(torch.int16), original))
            event = json.loads(Path(command["event_path"]).read_text())
            self.assertEqual(event["role"], "draft")
            self.assertEqual(event["input_token_id"], 20)
            self.assertEqual(event["logical_token_position"], 1)
            self.assertFalse(event["target_parameters_modified"])
            self.assertFalse(Path(self.config["arm_path"]).exists())
            # Later calls and requests consume clean inputs; no persistent parameter change.
            self.assertTrue(torch.equal(self.model(self.tokens, self.positions, self.batch)[0], expected))
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            states = [e for e in events if e["kind"] == "draft_path_state"]
            self.assertEqual({e["point"] for e in states}, set(fault.POINTS))
            self.assertEqual(sum(e["fault_applied"] for e in states), 1)
            for name, p in self.model.named_parameters():
                self.assertTrue(torch.equal(p, parameters[name]))

    def test_occurrence_and_phase_choose_only_eligible_forward(self):
        command, _ = self.arm("decoder_down_proj", phase="draft_decode", occurrence=2)
        self.model(self.tokens, self.positions, self.batch)  # Wrong phase.
        self.batch.forward_mode.name = "DECODE"
        self.batch.spec_info.hidden_states = self.features[:, :4]
        self.model(self.tokens, self.positions, self.batch)  # First eligible.
        self.assertFalse(Path(command["event_path"]).exists())
        self.model(self.tokens, self.positions, self.batch)  # Second eligible.
        self.assertEqual(json.loads(Path(command["event_path"]).read_text())["eligible_occurrence"], 2)

    def test_wrong_request_rejected_without_mutating_input(self):
        command, _ = self.arm("interface")
        command["rid"] = "wrong"
        client.write_json(Path(self.config["arm_path"]), command)
        original = self.features.clone()
        with self.assertRaisesRegex(RuntimeError, "different active request"):
            self.model(self.tokens, self.positions, self.batch)
        self.assertTrue(torch.equal(original, self.features))
        self.assertFalse(Path(command["event_path"]).exists())

    def test_target_installer_is_noop_even_when_armed(self):
        command, _ = self.arm("interface")
        target = Draft(layers=36)
        target.register_forward_hook(fault.create_draft_fault_hook(self.config))
        first = target(self.tokens, self.positions, self.batch)[0]
        second = target(self.tokens, self.positions, self.batch)[0]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(target.fc._forward_pre_hooks), 0)
        self.assertFalse(Path(command["event_path"]).exists())
        self.assertTrue(Path(self.config["arm_path"]).exists())

    def test_invalid_decode_and_bounds_rejected(self):
        command, _ = self.arm("interface")
        for point in ("interface", "fusion"):
            command.update(point=point, phase="draft_decode")
            with self.assertRaisesRegex(ValueError, "bypasses"):
                fault.validate_command(command)
        with self.assertRaises(IndexError):
            fault.copy_and_flip(self.features, {"point": "fusion", "row": 2,
                                               "bit_positions": [7]}, 4)
        with self.assertRaises(TypeError):
            fault.copy_and_flip(self.features.float(), {"point": "fusion"}, 4)

    def test_target_prefill_comparison_excludes_changed_candidate_tree(self):
        paths = [self.root / f"2026-10-09_{name}.jsonl" for name in ("clean", "fault")]
        for index, path in enumerate(paths):
            events = [{"kind": kind, "role": "target", "forward_mode": "EXTEND",
                       "rid": str(index), "worker_pid": index, "timestamp_ns": index,
                       "hidden": {"slice_values": [[1.]]}}
                      for kind in ("intermediate", "model_hidden", "model_logits")]
            events.append({"kind": "model_hidden", "role": "target",
                           "forward_mode": "TARGET_VERIFY", "value": index})
            path.write_text("\n".join(json.dumps(e) for e in events))
        self.assertEqual(client.target_first_prefill(paths[0]), client.target_first_prefill(paths[1]))
        paths[1].write_text(paths[1].read_text().replace("1.0", "2.0"))
        self.assertNotEqual(client.target_first_prefill(paths[0]), client.target_first_prefill(paths[1]))


if __name__ == "__main__":
    unittest.main()
