"""CPU checks for watchdog isolation and failed-request metric denominators."""
import importlib
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

c = importlib.import_module("experiments.2026-10-10_sglang_kv_campaign")


class CampaignTests(unittest.TestCase):
    def test_stall_requires_confirmed_injection_and_wall_limit_remains_separate(self):
        self.assertIsNone(c.watchdog_reason(20, 19, False, 15, 180))
        self.assertEqual(c.watchdog_reason(20, 19, True, 15, 180), "request_stall")
        self.assertEqual(c.watchdog_reason(180, 1, True, 15, 180), "wall_timeout")

    def test_watchdog_stops_server_and_unblocks_http_before_returning(self):
        stopped = threading.Event()
        server = NS(url="unused", stop_event=threading.Event(),
                    process=NS(pid=123, poll=lambda: None))
        reasons = []
        def stop(reason):
            reasons.append(reason); stopped.set()
        server.stop = stop
        def http(*args):
            if not stopped.wait(3):
                raise RuntimeError("test HTTP thread was not unblocked")
            raise ConnectionError("owned worker terminated")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = root / "2026-10-10_injected.json"; event.write_text("{}")
            with patch.object(c.client, "http", side_effect=http):
                response, failure = c.guarded_request(server, {}, root / "2026-10-10_trace.jsonl", event, .02, 1)
        self.assertIsNone(response)
        self.assertEqual(failure["reason"], "request_stall")
        self.assertEqual(reasons, ["request_stall"])

    def test_normal_http_completion_does_not_terminate_server(self):
        server = NS(url="unused", stop_event=threading.Event(),
                    process=NS(pid=123, poll=lambda: None), stop=lambda reason: self.fail(reason))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(c.client, "http", return_value={"text":"normal"}):
                response, failure = c.guarded_request(server, {}, Path(directory)/"2026-10-10_trace.jsonl", None, 15, 180)
        self.assertEqual(response["text"], "normal")
        self.assertIsNone(failure)

    def test_stop_only_terminates_the_process_group_it_created(self):
        with tempfile.TemporaryDirectory() as directory:
            owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            server = c.Server(0, {}, Path(directory), threading.Event())
            server.process = owned
            server.log_handle = (Path(directory)/"2026-10-10_server.log").open("x")
            try:
                server.stop("cpu_ownership_check")
                self.assertIsNotNone(owned.poll())
                self.assertIsNone(other.poll())
                self.assertIsNone(server.process)
            finally:
                if owned.poll() is None:
                    owned.kill(); owned.wait()
                other.terminate(); other.wait()

    def test_failed_request_is_not_fabricated_as_completed_wrong_answer(self):
        meta = {"spec_num_proposed_drafts": 10, "spec_num_correct_drafts": 5,
                "finish_reason": {"type":"stop"}, "e2e_latency":1}
        normal = {"sample_id":0,"answer_correct":True,"predicted_number":"1",
                  "injection_checks_pass":True,"response":{"meta_info":meta,"output_ids":[1]}}
        failed = {**normal,"response":None,"answer_correct":False,"predicted_number":None,
                  "request_failure":{"reason":"request_stall"}}
        summary = c.summarize([failed], [normal])
        self.assertEqual(summary["request_stalls"], 1)
        self.assertEqual(summary["correct_to_wrong_completed"], 0)
        self.assertEqual(summary["baseline_correct_to_request_failure"], 1)
        self.assertEqual(summary["task_success_rate_all_requests"], 0)
        self.assertIsNone(summary["accuracy_completed_only"])

    def test_lane_control_files_are_independent_without_mutating_reference(self):
        reference = {"context_path":"old", "kv_fault_arm_path":"old_arm",
                     "hooks":[{"config":{"context_path":"old","arm_path":"old_arm"}}]}
        a = c.make_lane_manifest(reference, Path("gpu0"))
        b = c.make_lane_manifest(reference, Path("gpu2"))
        self.assertNotEqual(a["context_path"], b["context_path"])
        self.assertEqual(reference["context_path"], "old")
        self.assertEqual(a["hooks"][0]["config"]["arm_path"], a["kv_fault_arm_path"])


if __name__ == "__main__":
    unittest.main()
