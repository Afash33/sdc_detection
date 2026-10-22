"""CPU tests of dtype correctness, actual consumed values and request lifecycle."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

fault = importlib.import_module("fault_injection.2026-10-09_sglang_target_output_bitflip")


class Processor(nn.Module):
    def __init__(self, vocab=151936):
        super().__init__()
        self.vocab_size = vocab

    def _compute_lm_head(self, hidden, head, bias=None):
        return hidden @ head.T

    def _get_logits(self, hidden, head, metadata):
        return self._compute_lm_head(hidden, head).float()

    def forward(self, ids, hidden, head, batch):
        logits = self._get_logits(hidden, head, batch)
        return SimpleNamespace(next_token_logits=logits, logprobs=torch.log_softmax(logits, -1))


class Target(nn.Module):
    def __init__(self, layers=36):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers)
        self.layers = nn.ModuleList()
        self.norm = nn.Identity()

    def forward(self, tokens, positions, batch):
        return self.norm(batch.hidden), [batch.hidden.clone()] * 3


class OutputFaultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = {"context_path": str(self.root / "2026-10-09_context.json"),
                       "arm_path": str(self.root / "2026-10-09_arm.json")}
        self.batch = SimpleNamespace(hidden=torch.tensor([[1.5, 2.]], dtype=torch.bfloat16),
            forward_mode=SimpleNamespace(name="EXTEND", is_idle=lambda: False))
        self.tokens, self.positions = torch.tensor([1]), torch.tensor([0])
        self.head = torch.eye(2, dtype=torch.bfloat16)
        self.target, self.processor = Target(), Processor()
        for module in (self.target, self.processor):
            module.register_forward_hook(fault.create_target_output_fault_hook(self.config))
        self.target(self.tokens, self.positions, self.batch)
        self.processor(self.tokens, self.batch.hidden, self.head, self.batch)

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, point, bit, *, phase="prefill", occurrence=1):
        trace = self.root / "2026-10-09_trace.jsonl"
        Path(self.config["context_path"]).write_text(json.dumps(
            {"rid": "request", "sample_id": 0, "trace_path": str(trace)}))
        command = {"point": point, "phase": phase, "fault_id": point, "rid": "request",
            "event_path": str(self.root / "2026-10-09_event.json"), "bit_positions": [bit],
            "row": -1, "channel": 0, "occurrence": occurrence}
        Path(self.config["arm_path"]).write_text(json.dumps(command))
        return command

    def test_fp32_flips_actual_bit30_and_preserves_original(self):
        x = torch.tensor([[1.5, 2.]], dtype=torch.float32)
        original = x.view(torch.int32).clone()
        changed, e = fault.copy_and_flip(x, {"point": "logits", "bit_positions": [30]})
        self.assertTrue(torch.isnan(changed[0, 0]))
        self.assertEqual(e["before_bits_hex"], "0x3fc00000")
        self.assertEqual(e["after_bits_hex"], "0x7fc00000")
        self.assertTrue(torch.equal(original, x.view(torch.int32)))
        json.dumps(e, allow_nan=False)

    def test_bf16_and_fp32_wrong_dtype_or_bit_rejected(self):
        for point, bit in (("lm_head", 30), ("logits", 32)):
            with self.assertRaises(ValueError):
                fault.validate_command({"point": point, "phase": "prefill", "fault_id": "x",
                    "rid": "x", "event_path": "x", "bit_positions": [bit]})
        with self.assertRaises(TypeError):
            fault.copy_and_flip(self.batch.hidden, {"point": "logits"})

    def test_norm_replaces_output_without_changing_auxiliary_or_next_request(self):
        command = self.arm("final_norm", 7)
        original = self.batch.hidden.clone()
        output, aux = self.target(self.tokens, self.positions, self.batch)
        self.assertFalse(torch.equal(output, original))
        self.assertTrue(all(torch.equal(a, original) for a in aux))
        self.assertTrue(torch.equal(self.batch.hidden, original))
        self.assertTrue(torch.equal(self.target(self.tokens, self.positions, self.batch)[0], original))
        self.assertTrue(Path(command["event_path"]).exists())

    def test_lm_head_and_logits_changes_reach_logprobs_and_fire_once(self):
        for point, bit in (("lm_head", 15), ("logits", 31)):
            with self.subTest(point=point):
                (self.root / "2026-10-09_event.json").unlink(missing_ok=True)
                command = self.arm(point, bit)
                result = self.processor(self.tokens, self.batch.hidden, self.head, self.batch)
                self.assertEqual(float(result.next_token_logits[0, 0]), -1.5)
                self.assertTrue(torch.equal(result.logprobs, torch.log_softmax(result.next_token_logits, -1)))
                later = self.processor(self.tokens, self.batch.hidden, self.head, self.batch)
                self.assertEqual(float(later.next_token_logits[0, 0]), 1.5)
                e = json.loads(Path(command["event_path"]).read_text())
                self.assertFalse(e["parameters_modified"])
                self.assertFalse(Path(self.config["arm_path"]).exists())

    def test_phase_and_occurrence_only_count_eligible_calls(self):
        command = self.arm("logits", 31, phase="target_verify", occurrence=2)
        self.processor(self.tokens, self.batch.hidden, self.head, self.batch)
        self.batch.forward_mode.name = "TARGET_VERIFY"
        self.processor(self.tokens, self.batch.hidden, self.head, self.batch)
        self.assertFalse(Path(command["event_path"]).exists())
        self.processor(self.tokens, self.batch.hidden, self.head, self.batch)
        self.assertEqual(json.loads(Path(command["event_path"]).read_text())["eligible_occurrence"], 2)

    def test_draft_processor_is_not_wrapped(self):
        draft = Processor(32000)
        draft.register_forward_hook(fault.create_target_output_fault_hook(self.config))
        self.arm("logits", 31)
        output = draft(self.tokens, self.batch.hidden, self.head, self.batch)
        self.assertEqual(float(output.next_token_logits[0, 0]), 1.5)
        self.assertNotIn("_get_logits", draft.__dict__)

    def test_wrong_rid_rejected(self):
        self.arm("logits", 31)
        path = Path(self.config["arm_path"])
        command = json.loads(path.read_text()); command["rid"] = "other"
        path.write_text(json.dumps(command))
        with self.assertRaisesRegex(RuntimeError, "different request"):
            self.processor(self.tokens, self.batch.hidden, self.head, self.batch)

    def test_shared_factory_installs_both_model_and_processor(self):
        model, processor = Target(), Processor()
        shared = fault.create_target_output_fault_hook(self.config)
        model.register_forward_hook(shared)
        processor.register_forward_hook(shared)
        model(self.tokens, self.positions, self.batch)
        processor(self.tokens, self.batch.hidden, self.head, self.batch)
        self.arm("logits", 31)
        result = processor(self.tokens, self.batch.hidden, self.head, self.batch)
        self.assertEqual(float(result.next_token_logits[0, 0]), -1.5)

    def test_client_support_keeps_old_hooks_and_records_dependencies(self):
        client = importlib.import_module("experiments.2026-10-06_sglang_baseline")
        old_hook = {"name": "original", "config": {"context_path": "context"}}
        manifest = {"context_path": "context", "hooks": [old_hook]}
        client.add_target_output_support(manifest, self.root, "2026-10-09")
        self.assertEqual(manifest["hooks"][1], old_hook)
        self.assertEqual(manifest["hooks"][0]["target_modules"], ["model", "logits_processor"])
        self.assertTrue(manifest["telemetry_dependencies"])
        client.add_target_output_support(manifest, self.root, "2026-10-09")
        self.assertEqual(len(manifest["hooks"]), 2)

    def test_client_rejects_missing_bit_evidence(self):
        client = importlib.import_module("experiments.2026-10-06_sglang_baseline")
        record = {"sample_id": 0, "request": {"rid": "request"},
            "target_output_fault_support": True,
            "response": {"output_ids": [], "meta_info": {"output_token_logprobs": [],
                "output_top_logprobs": [], "cached_tokens": 0, "spec_verify_ct": 0}}}
        trace = self.root / "2026-10-09_invalid_trace.jsonl"
        trace.write_text(json.dumps({"kind": "target_output_state", "point": "logits",
            "role": "target", "fault_applied": True, "rid": "request", "sample_id": 0}) + "\n")
        errors, _ = client.check_sample(record, trace)
        self.assertIn("unexpected Target output injection in clean run", errors)


if __name__ == "__main__":
    unittest.main()
