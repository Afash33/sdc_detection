"""Full KV fault campaign: owned servers, request watchdog, isolation and comparison.

Reuse existing injection and full telemetry without changing SGLang kernels.
One request at a time per A100; failed requests are never retried as successes.
Restart is between experiments, not recovery of the corrupted inference.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import hashlib
import importlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
client = importlib.import_module("experiments.2026-10-06_sglang_baseline")
kv = importlib.import_module("fault_injection.2026-10-10_sglang_kv_bitflip")
DATE = "2026-10-10"
PYTHON = "/data6t/hpk/sglang_env/bin/python"


class ResourceBusy(RuntimeError):
    """External GPU use; no experimental request has failed."""


def save(path, value):
    """Atomic checkpoints; sample payloads themselves are never overwritten."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    client.write_json(tmp, value)
    tmp.replace(path)


def read_events(path, partial=False):
    events, incomplete = [], 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if not partial:
                    raise
                incomplete += 1
    return events, incomplete


def response_equal(a, b):
    keys = [k for k in a["meta_info"] if k.startswith("spec_")]
    return (a["output_ids"] == b["output_ids"]
            and a["meta_info"]["output_token_logprobs"] == b["meta_info"]["output_token_logprobs"]
            and all(a["meta_info"][k] == b["meta_info"][k] for k in keys))


def watchdog_reason(elapsed, quiet, injected, stall_seconds, wall_seconds):
    if injected and quiet >= stall_seconds:
        return "request_stall"
    if elapsed >= wall_seconds:
        return "wall_timeout"
    return None


class Server:
    def __init__(self, gpu, manifest, control_dir, stop_event):
        self.gpu, self.manifest, self.control_dir = gpu, manifest, control_dir
        self.stop_event = stop_event
        self.port = 30000 + gpu
        self.url = f"http://127.0.0.1:{self.port}"
        self.process, self.log_handle, self.log_path = None, None, None
        self.serial = len(list(control_dir.glob(f"{DATE}_server_*_process.json")))

    def start(self):
        if self.process is not None:
            raise RuntimeError("stop owned server before restarting")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"port {self.port} already occupied; will not touch existing service")
        self.serial += 1
        self.log_path = self.control_dir / f"{DATE}_server_{self.serial:03d}.log"
        self.log_handle = self.log_path.open("x")
        m = self.manifest
        command = ["/data6t/hpk/scripts/run_sdc_experiment.sh", "sglang", str(self.gpu), "--",
            "python", "-u", "-m", "sglang.launch_server", "--model-path", m["target"],
            "--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", m["draft"],
            "--speculative-num-steps", "3", "--speculative-eagle-topk", "4",
            "--speculative-num-draft-tokens", "16", "--dtype", "bfloat16",
            "--context-length", str(m["context_length"]), "--mem-fraction-static", "0.75",
            "--max-running-requests", "1", "--random-seed", "0", "--enable-deterministic-inference",
            "--disable-radix-cache", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
            "--forward-hooks", json.dumps(m["hooks"]), "--host", "127.0.0.1", "--port", str(self.port)]
        self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.log_handle,
            stderr=subprocess.STDOUT, start_new_session=True)
        save(self.control_dir / f"{DATE}_server_{self.serial:03d}_process.json",
             {"pid": self.process.pid, "gpu": self.gpu, "command": command,
              "log": str(self.log_path), "started_at": time.time()})
        start = time.monotonic()
        while time.monotonic() - start < 240:
            if self.stop_event.is_set():
                raise InterruptedError("campaign stopped during startup")
            if self.process.poll() is not None:
                if "is not idle" in self.log_path.read_text():
                    raise ResourceBusy(f"A100 GPU {self.gpu} is occupied; no existing process was touched")
                raise RuntimeError(f"owned server startup failed: {self.log_path}")
            if "The server is fired up and ready to roll!" in self.log_path.read_text():
                info = client.http(self.url, "/server_info", timeout=5)
                client.validate_server(info, m)
                save(self.control_dir / f"{DATE}_server_{self.serial:03d}_info.json", info)
                return
            time.sleep(0.5)
        raise TimeoutError(f"server warmup exceeded 240s: {self.log_path}")

    def stop(self, reason="normal_shutdown"):
        p = self.process
        if p is None:
            return
        # Only the process group created by this instance is ever signalled.
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait(timeout=15)
        self.log_handle.close()
        save(self.control_dir / f"{DATE}_server_{self.serial:03d}_shutdown.json",
             {"pid": p.pid, "returncode": p.returncode, "reason": reason, "stopped_at": time.time()})
        self.process = None
        # Let CUDA release the terminated worker's allocations before harness checks.
        time.sleep(1)


def guarded_request(server, request, trace, event_path, stall_seconds, wall_seconds):
    start = last_progress = time.monotonic()
    previous_size = -1
    reason = None
    executor = futures.ThreadPoolExecutor(max_workers=1)
    task = executor.submit(client.http, server.url, "/generate", request, wall_seconds + 10)
    try:
        while not task.done():
            now = time.monotonic()
            size = trace.stat().st_size if trace.exists() else 0
            if size != previous_size:
                previous_size, last_progress = size, now
            if server.stop_event.is_set():
                raise InterruptedError("campaign stopped during request")
            reason = watchdog_reason(now - start, now - last_progress,
                bool(event_path and event_path.exists()), stall_seconds, wall_seconds)
            if reason or server.process.poll() is not None:
                reason = reason or "server_error"
                break
            time.sleep(0.25)
        if reason:
            evidence = {"reason": reason, "elapsed_seconds": time.monotonic() - start,
                        "quiet_seconds": time.monotonic() - last_progress,
                        "trace_bytes": previous_size, "server_pid": server.process.pid}
            server.stop(reason)
            try:
                task.result(timeout=15)
            except Exception:
                pass
            return None, evidence
        try:
            return task.result(), None
        except Exception as exc:
            server.stop("request_exception")
            return None, {"reason": "server_error", "error": repr(exc),
                          "elapsed_seconds": time.monotonic() - start}
    finally:
        if not task.done():
            server.stop("request_cleanup")
        executor.shutdown(wait=True)


def execute_sample(server, directory, sample, fault, args):
    directory.mkdir(parents=True, exist_ok=True)
    index = sample["sample_id"]
    final_path = directory / f"{DATE}_sample_{index:03d}.json"
    if final_path.exists():
        return json.loads(final_path.read_text())
    artifact_dir = directory
    trace_name = f"{DATE}_trace_sample_{index:03d}.jsonl"
    attempt = 1
    while (artifact_dir / trace_name).exists():
        attempt += 1
        artifact_dir = directory / f"{DATE}_attempt_sample_{index:03d}_{attempt:02d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace = artifact_dir / trace_name
    event_path = artifact_dir / f"{DATE}_kv_fault_sample_{index:03d}_event.json" if fault else None
    sample = copy.deepcopy(sample)
    sample["request"]["rid"] = f"{DATE}_{directory.name}_{index:03d}_attempt{attempt:02d}"
    context = Path(server.manifest["context_path"])
    arm = Path(server.manifest["kv_fault_arm_path"])
    if context.exists() or arm.exists():
        raise RuntimeError("stale context/arm: refusing to overwrite active request")
    save(context, {"sample_id": index, "rid": sample["request"]["rid"], "trace_path": str(trace)})
    if fault:
        command = {**fault, "fault_id": "kv_" + hashlib.sha256(sample["request"]["rid"].encode()).hexdigest()[:16],
                   "rid": sample["request"]["rid"], "event_path": str(event_path)}
        kv.validate_command(command)
        save(arm, command)
    start = time.monotonic()
    try:
        response, failure = guarded_request(server, sample["request"], trace, event_path,
                                            args.stall_seconds, args.wall_seconds)
    finally:
        # On failure guarded_request stops the worker BEFORE removing control files.
        context.unlink(missing_ok=True)
        arm.unlink(missing_ok=True)
    record = {**sample, "response": response, "request_failure": failure,
              "trace_path": str(trace), "physical_gpu": server.gpu,
              "server_generation": server.serial, "client_elapsed_seconds": time.monotonic() - start,
              "kv_recording": True, "kv_fault_support": True, "fault_context": None,
              "predicted_number": client.answer_number(response["text"]) if response else None}
    if fault:
        if not event_path.is_file():
            raise RuntimeError(f"request failed before confirmed injection; infrastructure error: {trace}")
        record["kv_fault_event"] = json.loads(event_path.read_text())
    record["answer_correct"] = record["predicted_number"] is not None and record["predicted_number"] == record["gold_number"]
    events, incomplete = read_events(trace, partial=bool(failure))
    if failure:
        issues = kv.check_events(record, events)
        # Kernel may stall before candidate verifier lifetime can be observed.
        pending = "missing candidate KV lifetime decision"
        fatal = [issue for issue in issues if issue != pending]
        if fatal:
            raise RuntimeError(f"failed request injection evidence invalid: {fatal}")
        record.update(injection_checks_pass=True, incomplete_trace_lines=incomplete,
                      pending_lifecycle=pending in issues, recording_checks_pass=None)
    else:
        issues, events = client.check_sample(record, trace)
        if issues:
            raise RuntimeError(f"completed request recording failed: {issues}")
        record.update(injection_checks_pass=True, recording_checks_pass=True)
    record["trace_events"] = len(events)
    with final_path.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return record


def make_lane_manifest(original, directory):
    m = copy.deepcopy(original)
    context = str(directory / f"{DATE}_active_sample.json")
    arm = str(directory / f"{DATE}_kv_fault_arm.json")
    m.update(context_path=context, kv_fault_arm_path=arm)
    for hook in m["hooks"]:
        hook["config"]["context_path"] = context
        if "arm_path" in hook["config"]:
            hook["config"]["arm_path"] = arm
    return m


def summarize(records, baseline=None):
    completed = [r for r in records if r["response"] is not None]
    proposed = sum(r["response"]["meta_info"]["spec_num_proposed_drafts"] for r in completed)
    correct = sum(r["response"]["meta_info"]["spec_num_correct_drafts"] for r in completed)
    result = {"samples_recorded": len(records), "responses_completed": len(completed),
        "request_failures": len(records) - len(completed),
        "request_stalls": sum((r.get("request_failure") or {}).get("reason") == "request_stall" for r in records),
        "correct_answers": sum(r["answer_correct"] for r in records),
        "task_success_rate_all_requests": sum(r["answer_correct"] for r in records) / len(records) if records else None,
        "accuracy_completed_only": sum(r["answer_correct"] for r in completed) / len(completed) if completed else None,
        "length_truncated": sum(r["response"]["meta_info"]["finish_reason"]["type"] == "length" for r in completed),
        "answer_extraction_failures_completed": sum(r["predicted_number"] is None for r in completed),
        "weighted_spec_accept_rate_completed": correct / proposed if proposed else None,
        "total_output_tokens_completed": sum(len(r["response"]["output_ids"]) for r in completed),
        "instrumented_latency_median_completed": statistics.median(r["response"]["meta_info"]["e2e_latency"] for r in completed) if completed else None,
        "recording_checks_pass": all(r["response"] is not None and r.get("recording_checks_pass", False) for r in records),
        "evidence_checks_pass": all(r["injection_checks_pass"] for r in records),
        "performance_benchmark": False}
    if baseline is not None:
        pairs = [(baseline[r["sample_id"]], r) for r in records]
        result.update(output_ids_changed_completed=sum(a["response"]["output_ids"] != b["response"]["output_ids"] for a,b in pairs if b["response"]),
            correct_to_wrong_completed=sum(a["answer_correct"] and not b["answer_correct"] for a,b in pairs if b["response"]),
            wrong_to_correct_completed=sum(not a["answer_correct"] and b["answer_correct"] for a,b in pairs if b["response"]),
            baseline_correct_to_request_failure=sum(a["answer_correct"] and b["response"] is None for a,b in pairs))
    return result


def paired_metrics(normal, trial):
    events, _ = read_events(Path(trial["trace_path"]), partial=trial["response"] is None)
    r = {"sample_id": trial["sample_id"], "request_failure": trial["request_failure"],
         "baseline_correct": normal["answer_correct"], "trial_correct": trial["answer_correct"],
         "injection": trial.get("kv_fault_event"), "nonfinite_logit_rows": {},
         "kv_lifecycle_first_events": [e for e in events if e["kind"] == "kv_cache_lifecycle"][:2]}
    for role in ("target", "draft"):
        e = [e for e in events if e["kind"] == "model_logits" and e["role"] == role]
        r["nonfinite_logit_rows"][role] = sum(sum(not v for v in x["probabilities"]["finite_rows"]) for x in e)
    r["logit_statistics"] = {}
    for role in ("target", "draft"):
        probabilities = [e["probabilities"] for e in events if e["kind"] == "model_logits" and e["role"] == role]
        entropy = [v for p in probabilities for v in p["entropy_nats"] if v is not None]
        r["logit_statistics"][role] = {"max_finite_entropy_nats": max(entropy, default=None),
            "finite_rows_with_zero_top1_top2_margin": sum(ok and margin == 0
                for p in probabilities for ok,margin in zip(p["finite_rows"], p["top1_top2_logit_margin"]))}
    reads = [e for e in events if e["kind"] == "kv_cache_read"]
    r["kv_visible_fault_read_calls"] = sum(bool(e["visible_query_rows"]) and e["bits_at_read"] == trial["kv_fault_event"]["after_bits_hex"] for e in reads)
    r["kv_first_visible_query_rows"] = reads[0]["visible_query_rows"] if reads else []
    r["generated_tokens_before_failure"] = max((e["generated_index_after"] for e in events if e["kind"] == "verify_block"), default=1)
    baseline_events, _ = read_events(Path(normal["trace_path"]))
    event = trial["kv_fault_event"]
    def fault_forward(es, kind, role, layer=None):
        return [e for e in es if e["kind"] == kind and e.get("role") == role
                and e.get("forward_index") == event["forward_index"]
                and (layer is None or e.get("layer_id", e.get("layer")) == layer)]
    normal_write = fault_forward(baseline_events, "kv_cache_write", event["role"], event["layer"])
    trial_write = fault_forward(events, "kv_cache_write", event["role"], event["layer"])
    aligned = bool(normal_write and trial_write and normal_write[0]["input_token_ids"] == trial_write[0]["input_token_ids"]
                   and normal_write[0]["positions"] == trial_write[0]["positions"])
    a = fault_forward(baseline_events, "kv_attention_state", event["role"], event["layer"])
    b = fault_forward(events, "kv_attention_state", event["role"], event["layer"])
    r["first_fault_attention"] = {"same_input_rows": aligned, "state_returned": bool(b)}
    if aligned and a and b:
        h, t = a[0]["hidden"], b[0]["hidden"]
        rows = []
        for i, (x,y) in enumerate(zip(h["slice_values"], t["slice_values"])):
            pairs = [(u,v) for u,v in zip(x,y) if u is not None and v is not None]
            distance = sum((u-v)**2 for u,v in pairs)**.5
            denominator = sum(u*u for u,v in pairs)**.5 * sum(v*v for u,v in pairs)**.5
            cosine = max(0., min(2., 1-sum(u*v for u,v in pairs)/denominator)) if denominator and len(pairs)==len(x) else None
            rows.append({"row": i,"slice_valid_elements":len(pairs),"l2_slice_finite_pairs":distance,
                         "cosine_distance_slice":cosine,"trial_nonfinite_full":t["nonfinite_count"][i],
                         "baseline_l2_norm_full_finite":h["l2_full"][i],"trial_l2_norm_full_finite":t["l2_full"][i]})
        r["first_fault_attention"].update(rows=rows,
            scope="same-input first fault forward; distances use 128-D slice, norms use full dimensions and exclude nonfinite values")
    normal_target = [e for e in baseline_events if e["kind"] == "kv_cache_write" and e["role"] == "target" and e["forward_index"]==1]
    trial_target = [e for e in events if e["kind"] == "kv_cache_write" and e["role"] == "target" and e["forward_index"]==1]
    same_inputs = bool(normal_target and trial_target and normal_target[0]["input_token_ids"]==trial_target[0]["input_token_ids"]
                       and normal_target[0]["positions"]==trial_target[0]["positions"])
    r["first_target_verify_kv"] = {"same_input_rows": same_inputs, "state_returned": bool(trial_target)}
    if same_inputs:
        nm = {e["layer_id"]:e for e in normal_target}
        r["first_target_verify_kv"]["changed_write_layers"] = [e["layer_id"] for e in trial_target
            if any(e[k]["sha256"] != nm[e["layer_id"]][k]["sha256"] for k in ("key","value"))]
    if trial["response"]:
        a,b = normal["response"], trial["response"]
        r.update(output_ids_exact=a["output_ids"]==b["output_ids"],
            logprobs_exact=a["meta_info"]["output_token_logprobs"]==b["meta_info"]["output_token_logprobs"],
            first_divergence=next((i for i,(x,y) in enumerate(zip(a["output_ids"],b["output_ids"])) if x!=y), min(len(a["output_ids"]),len(b["output_ids"])) if len(a["output_ids"])!=len(b["output_ids"]) else None),
            baseline_accept_rate=a["meta_info"]["spec_accept_rate"], trial_accept_rate=b["meta_info"]["spec_accept_rate"],
            token_id_zero_count=b["output_ids"].count(0))
    return r


def campaign(args):
    directory = args.run_dir.resolve()
    metadata_path = directory / f"{DATE}_campaign.json"
    stop_event, progress_lock = threading.Event(), threading.Lock()
    previous_term = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        stop_event.set()
        raise KeyboardInterrupt("campaign termination requested")
    signal.signal(signal.SIGTERM, terminate)
    servers = []
    if args.resume:
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_sha256"] != client.sha256(Path(__file__)):
            raise ValueError("campaign source changed; cannot silently resume")
        metadata.pop("error", None)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {"date": DATE, "status": "preparing", "source_sha256": client.sha256(Path(__file__)),
                    "num_samples": args.num_samples, "max_new_tokens": args.max_new_tokens,
                    "gpus": args.gpus, "bit": args.bit, "stall_seconds": args.stall_seconds,
                    "wall_seconds": args.wall_seconds, "groups": {}, "scope": "fixed one element per request; not random vulnerability estimate"}
        save(metadata_path, metadata)
    for key in ("num_samples", "max_new_tokens", "gpus", "bit", "stall_seconds", "wall_seconds"):
        if metadata[key] != getattr(args, key):
            raise ValueError(f"resume parameter differs: {key}")
    baseline_dir = directory / f"{DATE}_baseline"
    if not baseline_dir.exists():
        client.prepare(SimpleNamespace(run_dir=baseline_dir, reference_run=None, date=DATE,
            dataset=ROOT/"data_bundle/tinyGSM8k/main/test-00000-of-00001.parquet",
            target=Path("/data6t/hpk/models/Qwen3-8B"), draft=Path("/data6t/hpk/models/Qwen3-8B-eagle3"),
            num_samples=args.num_samples, max_new_tokens=args.max_new_tokens, context_length=4096,
            record_kv=True, kv_fault_support=True, draft_fault_support=False, target_output_fault_support=False))
    manifest_path, manifest = client.read_run(baseline_dir)
    for path, digest in manifest["telemetry_dependencies"].items():
        if client.sha256(Path(path)) != digest:
            raise ValueError("frozen injection/telemetry dependency changed")
    samples = json.loads((baseline_dir / f"{DATE}_samples.json").read_text())
    baseline = []
    lane_manifest = {}
    for gpu in args.gpus:
        control = directory / f"{DATE}_gpu_{gpu}_control"
        control.mkdir(exist_ok=True)
        m = make_lane_manifest(manifest, control)
        lane_manifest[gpu] = m
        server = Server(gpu, m, control, stop_event)
        servers.append(server)
    try:
        if not all((baseline_dir/f"{DATE}_sample_{s['sample_id']:03d}.json").exists() for s in samples):
            server = servers[0]
            server.start()
            manifest.update(status="running", campaign=str(directory), hooks=lane_manifest[server.gpu]["hooks"],
                            context_path=server.manifest["context_path"], kv_fault_arm_path=server.manifest["kv_fault_arm_path"])
            save(manifest_path, manifest)
            for sample in samples:
                r = execute_sample(server, baseline_dir, sample, None, args)
                if not r["response"]:
                    raise RuntimeError("healthy baseline failed; stop rather than label as injected fault")
                baseline.append(r)
                save(directory/f"{DATE}_progress.json", {"phase":"baseline", "completed":len(baseline),"total":len(samples)})
                print(f"BASELINE {len(baseline)}/{len(samples)} tokens={len(r['response']['output_ids'])} correct={r['answer_correct']}",flush=True)
        else:
            baseline = [json.loads((baseline_dir/f"{DATE}_sample_{s['sample_id']:03d}.json").read_text()) for s in samples]
        manifest["status"] = "completed"
        save(manifest_path, manifest)
        save(baseline_dir/f"{DATE}_summary.json", summarize(baseline))
        metadata["baseline"] = summarize(baseline)
        metadata["status"] = "running_fault_groups"
        save(metadata_path, metadata)

        def probe(server, reason):
            path = server.control_dir / f"{DATE}_healthy_probe_server_{server.serial:03d}_{reason}"
            r = execute_sample(server, path, samples[0], None, args)
            exact = r["response"] is not None and response_equal(baseline[0]["response"], r["response"])
            if exact:
                exact = client.trace_fingerprint(Path(baseline[0]["trace_path"])) == client.trace_fingerprint(Path(r["trace_path"]))
            save(path/f"{DATE}_comparison.json", {"normal_output_logprobs_spec_and_recorded_computation_exact":exact})
            if not exact:
                raise RuntimeError("healthy probe differs from baseline; no next injection allowed")

        groups = [(role,region,component) for role in ("target","draft") for region in ("history","candidate") for component in ("key","value")]
        def worker(number, server):
            if server.process is None:
                server.start()
            probe(server,"startup")
            try:
                for group_index,(role,region,component) in enumerate(groups):
                    if group_index % len(servers) != number:
                        continue
                    if stop_event.is_set():
                        raise InterruptedError("other lane stopped")
                    name=f"{role}_{region}_{component}"
                    run_dir=directory/f"{DATE}_{name}"
                    run_dir.mkdir(exist_ok=True)
                    fault={"role":role,"region":region,"component":component,
                        "layer":20 if role=="target" else 0,"head":0,"dimension":0,
                        "history_position":-1,"candidate_row":1 if role=="target" else 0,
                        "bit_positions":[args.bit],"occurrence":1}
                    gm=copy.deepcopy(server.manifest)
                    gm.update(fault_injection=True,fault_config=fault,reference_run=str(baseline_dir),
                              num_samples=len(samples),physical_gpu=server.gpu,status="running")
                    save(run_dir/f"{DATE}_manifest.json",gm)
                    records=[]
                    for sample in samples:
                        if stop_event.is_set():
                            raise InterruptedError("campaign stopped")
                        r=execute_sample(server,run_dir,sample,fault,args)
                        records.append(r)
                        if r["response"] is None and server.process is None:
                            server.start()
                            probe(server,f"after_failure_{name}_{sample['sample_id']:03d}")
                        with progress_lock:
                            metadata["groups"][name]={**summarize(records,baseline),"gpu":server.gpu,"status":"running"}
                            save(metadata_path,metadata)
                        print(f"GROUP {name} {len(records)}/{len(samples)} completed={r['response'] is not None} correct={r['answer_correct']} gpu={server.gpu}",flush=True)
                    probe(server,f"after_group_{name}")
                    report=[paired_metrics(baseline[r['sample_id']],r) for r in records]
                    summary=summarize(records,baseline)
                    save(run_dir/f"{DATE}_summary.json",summary)
                    save(run_dir/f"{DATE}_paired_analysis.json",report)
                    gm["status"]="completed_including_request_failures"
                    save(run_dir/f"{DATE}_manifest.json",gm)
                    with progress_lock:
                        metadata["groups"][name]={**summary,"gpu":server.gpu,"status":"completed"}
                        save(metadata_path,metadata)
            except BaseException:
                stop_event.set()
                raise
            finally:
                server.stop()

        executor=futures.ThreadPoolExecutor(max_workers=len(servers))
        tasks=[executor.submit(worker,i,s) for i,s in enumerate(servers)]
        try:
            for task in futures.as_completed(tasks):
                task.result()
        finally:
            stop_event.set()
            executor.shutdown(wait=True)
        metadata["status"]="completed"
    except BaseException as exc:
        stop_event.set()
        metadata.update(status="waiting_for_gpu" if isinstance(exc, ResourceBusy) else "failed",error=repr(exc))
        raise
    finally:
        for server in servers:
            server.stop("campaign_cleanup")
        save(metadata_path,metadata)
        cleanup={"owned_servers_stopped":all(s.process is None for s in servers),
                 "gpu_state":subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,utilization.gpu","--format=csv"],text=True),
                 "compute_processes":subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv"],text=True)}
        save(directory/f"{DATE}_cleanup.json",cleanup)
        signal.signal(signal.SIGTERM, previous_term)
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--gpus",type=int,nargs="+",choices=(0,2),default=[0,2])
    parser.add_argument("--num-samples",type=int,default=100)
    parser.add_argument("--max-new-tokens",type=int,default=1024)
    parser.add_argument("--bit",type=int,choices=range(16),default=14)
    parser.add_argument("--stall-seconds",type=float,default=15)
    parser.add_argument("--wall-seconds",type=float,default=180)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or args.stall_seconds<=0 or args.wall_seconds<=args.stall_seconds:
        parser.error("GPUs must be distinct; 0 < stall-seconds < wall-seconds required")
    campaign(args)


if __name__=="__main__":
    main()
