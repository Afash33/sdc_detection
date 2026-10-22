"""CPU checks for telemetry correctness; these do not validate GPU integration."""
import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

trace = importlib.import_module("experiments.2026-10-06_sglang_baseline_trace")
client = importlib.import_module("experiments.2026-10-06_sglang_baseline")


class BaselineRecordingTests(unittest.TestCase):
    def test_weight_evidence_checks_xor_and_exact_restore(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            injected_path = root / "2026-10-07_injected.json"
            restored_path = root / "2026-10-07_restored.json"
            event = {"fault_kind": "weight_bitflip", "status": "injected", "fault_id": "cpu",
                     "parameter_dtype": "torch.bfloat16", "worker_pid": 99,
                     "before_bits_hex": "0x3b6b", "after_bits_hex": "0x7b6b", "xor_mask_hex": "0x4000"}
            client.write_json(injected_path, event)
            self.assertTrue(client.read_weight_context(injected_path)["active"])
            restored = {"fault_kind": "weight_bitflip", "status": "restored", "fault_id": "cpu",
                        "worker_pid": 99, "restored_bits_hex": "0x3b6b", "fault_bits_hex": "0x7b6b",
                        "injected_event_path": str(injected_path)}
            client.write_json(restored_path, restored)
            self.assertFalse(client.read_weight_context(restored_path)["active"])
            restored["restored_bits_hex"] = "0x3b6a"
            client.write_json(restored_path, restored)
            with self.assertRaisesRegex(ValueError, "restoration evidence"):
                client.read_weight_context(restored_path)
            event["after_bits_hex"] = "0x7b6a"
            client.write_json(injected_path, event)
            with self.assertRaisesRegex(ValueError, "bitwise XOR"):
                client.read_weight_context(injected_path)

    def test_trace_comparison_ignores_bookkeeping_but_preserves_numeric_changes(self):
        with tempfile.TemporaryDirectory() as root:
            a = Path(root) / "2026-10-07_reference_trace.jsonl"
            b = Path(root) / "2026-10-07_repeat_trace.jsonl"
            event = {"kind": "model_hidden", "sample_id": 0, "slice_values": [[1., 2.]],
                     "rid": "old", "worker_pid": 1, "timestamp_ns": 10, "forward_index": 3}
            a.write_text(json.dumps(event) + "\n")
            event.update(rid="new", worker_pid=2, timestamp_ns=20, forward_index=99)
            b.write_text(json.dumps(event) + "\n")
            self.assertEqual(client.trace_fingerprint(a), client.trace_fingerprint(b))
            event["slice_values"][0][1] = 2.001
            b.write_text(json.dumps(event) + "\n")
            self.assertNotEqual(client.trace_fingerprint(a), client.trace_fingerprint(b))

    def test_repeat_preparation_preserves_reference_and_reuses_hook_context(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            reference, repeat = root / "reference", root / "repeat"
            reference.mkdir()
            for role in ("target", "draft"):
                (root / role).mkdir()
                (root / role / "config.json").write_text("{}")
            source = root / "2026-10-07_fake_telemetry.py"
            source.write_text("# fixture")
            manifest = {
                "date": "2026-10-07", "status": "completed", "num_samples": 1,
                "telemetry_source": str(source), "telemetry_source_sha256": client.sha256(source),
                "packages": {}, "target": str(root / "target"), "draft": str(root / "draft"),
                "model_config_sha256": {role: client.sha256(root / role / "config.json") for role in ("target", "draft")},
                "hooks": [{"config": {"context_path": str(reference / "2026-10-07_active_sample.json")}}],
                "context_path": str(reference / "2026-10-07_active_sample.json"),
            }
            samples = [{"sample_id": 0, "request": {"rid": "reference", "text": "same prompt",
                        "sampling_params": {"temperature": 0}}}]
            client.write_json(reference / "2026-10-07_manifest.json", manifest)
            client.write_json(reference / "2026-10-07_samples.json", samples)
            client.write_json(reference / "2026-10-07_summary.json", {"recording_checks_pass": True})
            client.prepare_repeat(SimpleNamespace(run_dir=repeat, reference_run=reference,
                                                  num_samples=None, date="2026-10-07"))
            _, result = client.read_run(repeat)
            self.assertEqual(result["hooks"], manifest["hooks"])
            self.assertEqual(result["context_path"], manifest["context_path"])
            self.assertEqual(result["status"], "prepared_not_run")
            self.assertEqual(json.loads((reference / "2026-10-07_samples.json").read_text()), samples)
            fresh = json.loads((repeat / "2026-10-07_samples.json").read_text())
            self.assertEqual(fresh[0]["request"]["text"], samples[0]["request"]["text"])
            self.assertNotEqual(fresh[0]["request"]["rid"], samples[0]["request"]["rid"])

    def test_verify_rows_are_not_treated_as_prefill(self):
        batch = SimpleNamespace(forward_mode=SimpleNamespace(name="TARGET_VERIFY"))
        self.assertFalse(trace.prefill_last_only(batch, 16))
        batch.forward_mode.name = "DRAFT_EXTEND"
        self.assertFalse(trace.prefill_last_only(batch, 4))
        self.assertTrue(trace.prefill_last_only(batch, 1054))
        batch.forward_mode.name = "EXTEND"
        self.assertTrue(trace.prefill_last_only(batch, 1054))

    def test_hidden_norm_uses_all_dimensions_without_mutation(self):
        x = torch.tensor([[3., 4., 12.], [0., 0., 0.]], dtype=torch.bfloat16)
        bits = x.view(torch.int16).clone()
        d = trace.summarize_hidden(x, torch.tensor([9, 10]), slice_size=2)
        self.assertEqual(d["l2_full"], [13., 0.])
        self.assertEqual(d["positions"], [9, 10])
        self.assertEqual(d["slice_dimensions"], [0, 2])
        self.assertTrue(torch.equal(bits, x.view(torch.int16)))

    def test_large_and_nonfinite_values_are_serializable(self):
        x = torch.tensor([[1e36, -1e36], [float('nan'), float('inf')]])
        h = trace.summarize_hidden(x)
        p = trace.summarize_logits(x)
        self.assertTrue(math.isfinite(h["l2_full"][0]))
        self.assertEqual(h["nonfinite_count"], [0, 2])
        self.assertEqual(p["finite_rows"], [True, False])
        self.assertEqual(p["entropy_nats"][0], 0.)
        self.assertIsNone(p["entropy_nats"][1])
        json.dumps([h, p], allow_nan=False)

    def test_probabilities_use_whole_vocabulary(self):
        x = torch.zeros(2, 10)
        d = trace.summarize_logits(x, top_k=3)
        self.assertEqual(len(d["top_token_ids"][0]), 3)
        self.assertAlmostEqual(d["entropy_nats"][0], math.log(10))
        self.assertAlmostEqual(d["top_logprobs"][0][0], -math.log(10))

    def test_answer_extraction_does_not_use_arbitrary_reasoning_number(self):
        self.assertEqual(client.answer_number("Cost: 20+4+5\n#### $29.00"), "29")
        self.assertEqual(client.answer_number("#### 1,234"), "1234")
        self.assertIsNone(client.answer_number("The cost of chips is 3"))

    def test_checks_detect_missing_hidden_rows_and_wrong_committed_tokens(self):
        record = {"sample_id": 0, "request": {"rid": "cpu_test"},
                  "response": {"output_ids": [10, 11, 12], "meta_info": {
                      "output_token_logprobs": [[-.1, 10, None], [-.2, 11, None], [-.3, 12, None]],
                      "output_top_logprobs": [[], [], []], "cached_tokens": 0,
                      "spec_verify_ct": 1}}}
        events = []
        for role in ("target", "draft"):
            events += [{"kind": "model_hidden", "role": role, "forward_mode": "TARGET_VERIFY",
                        "auxiliary_hidden": [{}, {}, {}],
                        "final_hidden": {"row_indices": [0, 1], "original_shape": [2, 4]}},
                       {"kind": "model_logits", "role": role}]
        events += [{"kind": "intermediate"}, {
            "kind": "verify_block", "block_index": 0, "generated_index_before": 1,
            "generated_index_after": 3, "committed_token_ids": [11, 12],
            "accepted_row_indices": [0, 1], "target_probabilities": {"shape": [2, 8]}}]
        for e in events:
            e.update(sample_id=0, rid="cpu_test")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "2026-10-06_cpu_trace.jsonl"
            def save():
                path.write_text("\n".join(json.dumps(e) for e in events))
            save()
            self.assertEqual(client.check_sample(record, path)[0], [])
            events[0]["final_hidden"]["row_indices"] = [1]
            events[-1]["committed_token_ids"] = [11, 99]
            save()
            errors = client.check_sample(record, path)[0]
            self.assertIn("verification hidden rows were truncated", errors)
            self.assertIn("committed block tokens differ from final output", errors)


if __name__ == "__main__":
    unittest.main()
