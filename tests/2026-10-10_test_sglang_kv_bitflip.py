"""CPU checks for direct KV storage mutation, address selection and lifetime."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

fi = importlib.import_module("fault_injection.2026-10-10_sglang_kv_bitflip")


class Pool:
    def __init__(self):
        self.k = torch.ones(12, 2, 3, dtype=torch.bfloat16)
        self.v = torch.full_like(self.k, 2)

    def _get_key_buffer(self, layer):
        return self.k

    def _get_value_buffer(self, layer):
        return self.v

    def get_kv_buffer(self, layer):
        return self.k, self.v

    def set_kv_buffer(self, layer, loc, k, v):
        self.k[loc], self.v[loc] = k, v


def command(**kwargs):
    return dict(fault_id="cpu", rid="cpu", event_path="unused", role="target",
                region="history", component="key", layer=0, head=0, dimension=0,
                bit_positions=[14], occurrence=1, **kwargs)


class KVFaultTests(unittest.TestCase):
    def test_inplace_one_word_and_k_v_isolation(self):
        for component in ("key", "value"):
            pool = Pool()
            before_k, before_v = pool.k.clone(), pool.v.clone()
            cmd = command(); cmd["component"] = component
            event = fi.flip_storage(pool, 0, 4, cmd)
            self.assertEqual(int(event["before_bits_hex"], 16) ^ 0x4000,
                             int(event["after_bits_hex"], 16))
            self.assertEqual(int((pool.k.view(torch.int16) != before_k.view(torch.int16)).sum()),
                             1 if component == "key" else 0)
            self.assertEqual(int((pool.v.view(torch.int16) != before_v.view(torch.int16)).sum()),
                             1 if component == "value" else 0)
            with self.assertRaises(IndexError):
                fi.flip_storage(pool, 0, 0, cmd)

    def test_history_and_candidate_addresses(self):
        batch = NS(req_pool_indices=torch.tensor([0]), seq_lens=torch.tensor([3]),
                   positions=torch.tensor([3, 4]),
                   req_to_token_pool=NS(req_to_token=torch.tensor([[5, 7, 9, 1, 2]])))
        loc = torch.tensor([1, 2])
        self.assertEqual(fi.select_slot(batch, loc, command()), (9, None, 2))
        cmd = command(); cmd.update(region="candidate", candidate_row=1)
        self.assertEqual(fi.select_slot(batch, loc, cmd), (2, 1, 4))
        batch.req_to_token_pool.req_to_token[0, 2] = 1
        with self.assertRaisesRegex(ValueError, "overlaps"):
            fi.select_slot(batch, loc, command())

    def test_table_visibility_respects_lengths_and_branches(self):
        metadata = NS(page_table=torch.tensor([[5, 8, 9], [7, 9, 8]]),
                      cache_seqlens_int32=torch.tensor([2, 2]),
                      cu_seqlens_q=torch.tensor([0, 4, 5]))
        self.assertEqual(fi.visible_rows(metadata, 8), [0, 1, 2, 3])
        self.assertEqual(fi.visible_rows(metadata, 9), [4])

    def test_candidate_lifetime_retained_and_discarded(self):
        for retained in (True, False):
            session = fi.KVSession({}); session.rid = "cpu"
            cmd = command(); cmd.update(region="candidate", slot=4)
            session.watch = {"event": cmd, "context": {"rid": "cpu"}, "pool": Pool(), "valid": True}
            logged = []; session.log = lambda kind, **fields: logged.append(fields)
            session.verify_lifecycle(None, [4, 5], NS(accept_indices=torch.tensor([0 if retained else 1])))
            self.assertEqual(session.watch["valid"], retained)
            self.assertEqual(logged[0]["action"], "retained_as_history" if retained else "discarded")

    def test_write_gate_single_fire_and_recompute_ends_candidate_watch(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); arm = root / "2026-10-10_arm.json"
            event_path = root / "2026-10-10_event.json"
            trace = root / "2026-10-10_trace.jsonl"
            cmd = command(); cmd.update(role="draft", region="candidate", event_path=str(event_path))
            arm.write_text(json.dumps(cmd))
            session = fi.KVSession({"arm_path": str(arm)})
            target, draft = Pool(), Pool()
            session.install_pool(target, "target"); session.install_pool(draft, "draft")
            batch = NS(req_pool_indices=torch.tensor([0]), positions=torch.tensor([3]),
                       forward_mode=NS(name="DECODE"))
            context = {"rid": "cpu", "sample_id": 0, "trace_path": str(trace)}
            session.active[id(draft)] = {"batch": batch, "context": context, "index": 1}
            loc = torch.tensor([4]); k = torch.ones(1, 2, 3, dtype=torch.bfloat16)
            draft.set_kv_buffer(NS(layer_id=0), loc, k, k)
            self.assertFalse(arm.exists()); self.assertTrue(event_path.exists())
            self.assertTrue(session.watch["valid"])
            draft.set_kv_buffer(NS(layer_id=0), loc, k, k)
            self.assertFalse(session.watch["valid"])
            self.assertEqual(session.bits(draft, session.watch["event"]), "0x3f80")
            self.assertEqual(len([e for e in trace.read_text().splitlines() if '"fault_context"' in e]), 1)

    def test_invalid_bit_and_missing_support(self):
        cmd = command(); cmd["bit_positions"] = [16]
        with self.assertRaises(ValueError):
            fi.validate_command(cmd)
        client = importlib.import_module("experiments.2026-10-06_sglang_baseline")
        with self.assertRaisesRegex(ValueError, "record-kv"):
            client.add_kv_fault_support({}, Path("unused"), "2026-10-10")


if __name__ == "__main__":
    unittest.main()
