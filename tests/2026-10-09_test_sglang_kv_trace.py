"""CPU checks for actual-cache reads and causally aligned comparisons."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

kv = importlib.import_module("experiments.2026-10-09_sglang_kv_trace")
client = importlib.import_module("experiments.2026-10-06_sglang_baseline")


class KVRecordingTests(unittest.TestCase):
    def test_reads_written_storage_and_keeps_model_write_unchanged(self):
        class Pool:
            def __init__(self):
                self.k = torch.zeros(8, 1, 2, dtype=torch.bfloat16)
                self.v = self.k.clone()
                self.calls = 0

            def _get_key_buffer(self, layer):
                return self.k

            def _get_value_buffer(self, layer):
                return self.v

            def set_kv_buffer(self, layer, loc, k, v):
                self.calls += 1
                self.k[loc] = k * 2  # prove recorder reads storage, not input k
                self.v[loc] = v
                return "original_result"

        with tempfile.TemporaryDirectory() as root:
            pool = Pool()
            path = Path(root) / "2026-10-09_trace.jsonl"
            context = {"trace_path": str(path), "sample_id": 0, "rid": "cpu"}
            pool._sdc_kv_active = {"context": context, "role": "target", "index": 0,
                "mode": "EXTEND", "positions": [4, 5], "tokens": [10, 11],
                "artifact": Path(root) / "2026-10-09_kv.pt", "captures": []}
            kv._install_pool(pool)
            loc = torch.tensor([3, 6])
            k = torch.tensor([[[1., 2.]], [[3., 4.]]], dtype=torch.bfloat16)
            original = k.clone()
            result = pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, k, k)
            self.assertEqual(result, "original_result")
            self.assertEqual(pool.calls, 1)
            self.assertTrue(torch.equal(k, original))
            capture = pool._sdc_kv_active["captures"][0]
            self.assertTrue(torch.equal(capture["key"], k * 2))
            self.assertTrue(torch.equal(capture["value"], k))
            event = json.loads(path.read_text())
            self.assertEqual(event["cache_slots"], [3, 6])
            self.assertEqual(event["key"]["sha256"], kv.tensor_hash(k * 2))
            with self.assertRaisesRegex(ValueError, "dummy"):
                kv.read_written_rows(pool, 0, torch.tensor([0]))

    def test_comparison_excludes_different_input_suffix(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            reference, trial = root / "reference", root / "trial"
            reference.mkdir(); trial.mkdir()
            x = torch.ones(3, 1, 2, dtype=torch.bfloat16)
            for role in ("target", "draft"):
                a = {"role": role, "forward_index": 0, "positions": [0, 1, 2],
                     "input_token_ids": [1, 2, 3],
                     "captures": [{"layer_id": 0, "key": x, "value": x}]}
                torch.save(a, reference / f"2026-10-09_kv_sample_000_{role}_forward_000.pt")
                y = x.clone(); y[2] = 99
                a["input_token_ids"] = [1, 2, 4]
                a["captures"] = [{"layer_id": 0, "key": y, "value": y}]
                torch.save(a, trial / f"2026-10-09_kv_sample_000_{role}_forward_000.pt")
            output = root / "2026-10-09_comparison.json"
            kv.compare_runs(reference, [trial], output)
            report = json.loads(output.read_text())["reports"][0]
            for role in ("target", "draft"):
                self.assertEqual(report["first_prefill"][role]["compared_prefix_rows"], 2)
                self.assertTrue(report["first_prefill"][role]["layers"]["0"]["key"]["bitwise_equal"])

    def test_bit_comparison_detects_signed_zero_and_reports_nonfinite(self):
        a = torch.tensor([[[0., 1.]]], dtype=torch.bfloat16)
        b = torch.tensor([[[-0., 1.]]], dtype=torch.bfloat16)
        d = kv.aligned_difference(a, b)
        self.assertFalse(d["bitwise_equal"])
        self.assertEqual(d["l2_difference_finite"], 0.)
        b[0, 0, 1] = float("inf")
        d = kv.aligned_difference(a, b)
        self.assertEqual(d["trial_nonfinite_elements"], 1)
        json.dumps(d, allow_nan=False)

    def test_trace_hash_ignores_slots_but_preserves_kv_values(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = (Path(root) / name for name in ("2026-10-09_a.jsonl", "2026-10-09_b.jsonl"))
            e = {"kind": "kv_cache_write", "cache_slots": [1], "artifact_path": "old",
                 "role": "target", "layer_id": 20, "positions": [9], "key": {"sha256": "one"}}
            a.write_text(json.dumps(e) + "\n")
            e.update(cache_slots=[6], artifact_path="new")
            b.write_text(json.dumps(e) + "\n")
            self.assertEqual(client.trace_fingerprint(a), client.trace_fingerprint(b))
            e["key"]["sha256"] = "changed"
            b.write_text(json.dumps(e) + "\n")
            self.assertNotEqual(client.trace_fingerprint(a), client.trace_fingerprint(b))


if __name__ == "__main__":
    unittest.main()
