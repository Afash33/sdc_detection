"""Prepare, request, and check a normal tinyGSM8K EAGLE3 trace experiment.

This client does not start a server or select a GPU. With prepared Draft fault
support it can arm one activation bit flip per request in the worker hooks.
Use the existing server harness separately. Run with the sglang_env Python.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_target_output_support(manifest, run_dir, date):
    if manifest.get("target_output_fault_support"):
        return
    module = "fault_injection.2026-10-09_sglang_target_output_bitflip"
    arm = run_dir / f"{date}_target_output_fault_arm.json"
    # Run installer before ordinary telemetry, so telemetry sees the consumed value.
    manifest["hooks"].insert(0, {"name": "target_output_fault_and_trace",
        "target_modules": ["model", "logits_processor"],
        "hook_factory": module + ":create_target_output_fault_hook",
        "config": {"context_path": manifest["context_path"], "arm_path": str(arm)}})
    manifest["target_output_fault_support"] = True
    manifest["target_output_fault_arm_path"] = str(arm)
    dependencies = manifest.setdefault("telemetry_dependencies", {})
    for relative in ("fault_injection/2026-10-09_sglang_target_output_bitflip.py",
                     "fault_injection/2026-10-09_sglang_draft_bitflip.py",
                     "fault_injection/sglang_activation_bitflip.py",
                     "experiments/2026-10-06_sglang_baseline_trace.py"):
        dependencies[str(ROOT / relative)] = sha256(ROOT / relative)


def add_kv_fault_support(manifest, run_dir, date):
    if not manifest.get("kv_recording"):
        raise ValueError("KV fault support requires --record-kv")
    if manifest.get("kv_fault_support"):
        return
    module = "fault_injection.2026-10-10_sglang_kv_bitflip"
    arm = run_dir / f"{date}_kv_fault_arm.json"
    manifest["hooks"].insert(0, {"name": "kv_storage_fault_and_trace",
        "target_modules": ["model"], "hook_factory": module + ":create_kv_fault_hook",
        "config": {"context_path": manifest["context_path"], "arm_path": str(arm)}})
    manifest["kv_fault_support"] = True
    manifest["kv_fault_arm_path"] = str(arm)
    dependencies = manifest.setdefault("telemetry_dependencies", {})
    for relative in ("fault_injection/2026-10-10_sglang_kv_bitflip.py",
                     "fault_injection/2026-10-09_sglang_draft_bitflip.py",
                     "fault_injection/sglang_activation_bitflip.py",
                     "experiments/2026-10-06_sglang_baseline_trace.py"):
        dependencies[str(ROOT / relative)] = sha256(ROOT / relative)


def answer_number(text):
    # Strict task-format extraction. Missing #### is an extraction failure,
    # not an arbitrary last number from a chain of reasoning.
    matches = re.findall(r"####\s*([$]?\s*[+-]?\d[\d,]*(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace("$", "").replace(",", "").strip()).normalize())
    except InvalidOperation:
        return None


def prepare(args):
    if args.reference_run is not None:
        return prepare_repeat(args)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"will not overwrite existing run: {run_dir}")
    dataset = args.dataset.resolve()
    table = pq.read_table(dataset).to_pylist()
    count = args.num_samples if args.num_samples is not None else len(table)
    if not 1 <= count <= len(table):
        raise ValueError(f"num_samples must be between 1 and {len(table)}")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    samples = []
    for index, row in enumerate(table[:count]):
        input_ids = tokenizer.encode(row["input_formatted"], add_special_tokens=False)
        if len(input_ids) + args.max_new_tokens > args.context_length:
            raise ValueError(f"sample {index}: prompt+generation exceeds context_length")
        samples.append({
            "sample_id": index, "question": row["question"],
            "gold_answer": row["answer"], "gold_number": answer_number(row["answer"]),
            "input_ids": input_ids, "prompt_sha256": hashlib.sha256(
                row["input_formatted"].encode()).hexdigest(),
            "request": {
                "rid": f"{args.date}_normal_{run_dir.name}_{index:03d}",
                "text": row["input_formatted"],
                "sampling_params": {
                    "temperature": 0, "top_k": 1, "top_p": 1.0,
                    "repetition_penalty": 1.0, "max_new_tokens": args.max_new_tokens,
                    "stop": ["\n\nQuestion:", "\nQuestion:"],
                },
                "return_logprob": True, "top_logprobs_num": 8, "stream": False,
            },
        })
    run_dir.mkdir(parents=True)
    date = args.date
    context_path = run_dir / f"{date}_active_sample.json"
    config = {"context_path": str(context_path)}
    record_kv = getattr(args, "record_kv", False)
    telemetry_module = ("experiments.2026-10-09_sglang_kv_trace" if record_kv
                        else "experiments.2026-10-06_sglang_baseline_trace")
    telemetry_source = ROOT / (telemetry_module.replace(".", "/") + ".py")
    hooks = [{
        "name": "normal_model_hidden", "target_modules": ["model"],
        "hook_factory": telemetry_module + ":create_model_trace_hook",
        "config": config,
    }, {
        "name": "normal_model_logits", "target_modules": ["logits_processor"],
        "hook_factory": telemetry_module + ":create_logits_trace_hook",
        "config": config,
    }]
    manifest = {
        "date": date, "status": "prepared_not_run", "fault_injection": False,
        "dataset": str(dataset), "dataset_sha256": sha256(dataset),
        "dataset_total_rows": len(table), "num_samples": count,
        "selection": "first rows in original parquet order",
        "target": str(args.target.resolve()), "draft": str(args.draft.resolve()),
        "context_length": args.context_length, "max_new_tokens": args.max_new_tokens,
        "sampling": samples[0]["request"]["sampling_params"],
        "packages": {name: importlib.metadata.version(name)
                     for name in ("sglang", "torch", "transformers", "pyarrow")},
        "model_config_sha256": {role: sha256(path / "config.json")
                                for role, path in (("target", args.target), ("draft", args.draft))},
        "model_weight_identity": "local paths; config hashes only, no full weight hash",
        "server_required": {"algorithm": "EAGLE3", "draft_steps": 3, "tree_topk": 4,
                            "tree_tokens": 16, "tp": 1, "cuda_graph": False,
                            "radix_cache": False, "deterministic": True, "seed": 0},
        "context_path": str(context_path), "hooks": hooks,
        "telemetry_source": str(telemetry_source),
        "telemetry_source_sha256": sha256(telemetry_source),
        "limitations": [
            "instrumented eager run; latency includes synchronous collection overhead",
            "long prefill records only last hidden row per chunk; verification keeps all rows",
            "128 fixed dimensions are a slice, not a full hidden vector",
            "draft top IDs use its reduced 32000 vocabulary, not target token IDs",
            "accuracy is unweighted subset accuracy, not a full-GSM8K estimate",
            "provided few-shot input_formatted is preserved, including original example mistakes",
            "no attention matrices, full KV contents, full hidden vectors or full logits saved",
        ],
    }
    if record_kv:
        dependency = ROOT / "experiments/2026-10-06_sglang_baseline_trace.py"
        manifest["kv_recording"] = True
        manifest["telemetry_dependencies"] = {str(dependency): sha256(dependency)}
        manifest["limitations"][-1] = "full BF16 KV write rows saved; rejected/speculative rows are not necessarily committed history"
    if getattr(args, "draft_fault_support", False):
        draft_module = "fault_injection.2026-10-09_sglang_draft_bitflip"
        arm_path = run_dir / f"{date}_draft_fault_arm.json"
        hooks.append({"name": "draft_path_fault_and_trace", "target_modules": ["model"],
                      "hook_factory": draft_module + ":create_draft_fault_hook",
                      "config": {"context_path": str(context_path), "arm_path": str(arm_path)}})
        manifest["draft_fault_support"] = True
        manifest["draft_fault_arm_path"] = str(arm_path)
        dependencies = manifest.setdefault("telemetry_dependencies", {})
        for relative in ("fault_injection/2026-10-09_sglang_draft_bitflip.py",
                         "fault_injection/sglang_activation_bitflip.py",
                         "experiments/2026-10-06_sglang_baseline_trace.py"):
            dependencies[str(ROOT / relative)] = sha256(ROOT / relative)
    if getattr(args, "target_output_fault_support", False):
        add_target_output_support(manifest, run_dir, date)
    if getattr(args, "kv_fault_support", False):
        add_kv_fault_support(manifest, run_dir, date)
    write_json(run_dir / f"{date}_manifest.json", manifest)
    write_json(run_dir / f"{date}_samples.json", samples)
    write_json(run_dir / f"{date}_forward_hooks.json", hooks)
    print(f"Prepared {count}/{len(table)} samples: {run_dir}")
    print(f"max prompt tokens: {max(len(s['input_ids']) for s in samples)}")


def prepare_repeat(args):
    """Reuse frozen prompts and the server's existing context-file path."""
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"will not overwrite existing run: {run_dir}")
    _, original = read_run(args.reference_run)
    if original["status"] != "completed":
        raise ValueError("reference run must have completed successfully")
    summary = json.loads((args.reference_run / f"{original['date']}_summary.json").read_text())
    if not summary["recording_checks_pass"]:
        raise ValueError("reference recording checks did not pass")
    if sha256(Path(original["telemetry_source"])) != original["telemetry_source_sha256"]:
        raise ValueError("reference telemetry source has changed")
    for path, digest in original.get("telemetry_dependencies", {}).items():
        if sha256(Path(path)) != digest:
            raise ValueError("reference telemetry dependency has changed")
    for name, version in original["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"dependency version changed: {name}")
    for role in ("target", "draft"):
        if sha256(Path(original[role]) / "config.json") != original["model_config_sha256"][role]:
            raise ValueError(f"{role} config changed")
    samples = json.loads((args.reference_run / f"{original['date']}_samples.json").read_text())
    count = args.num_samples if args.num_samples is not None else len(samples)
    if not 1 <= count <= len(samples):
        raise ValueError("invalid repeat sample count")
    samples = samples[:count]
    for sample in samples:
        sample["request"]["rid"] = f"{args.date}_normal_{run_dir.name}_{sample['sample_id']:03d}"
    manifest = copy.deepcopy(original)
    for key in ("start_time", "end_time", "error", "resource_monitor_physical_gpu"):
        manifest.pop(key, None)
    manifest.update(date=args.date, status="prepared_not_run", num_samples=count,
                    reference_run=str(args.reference_run.resolve()))
    if getattr(args, "target_output_fault_support", False):
        add_target_output_support(manifest, run_dir, args.date)
    if getattr(args, "kv_fault_support", False):
        add_kv_fault_support(manifest, run_dir, args.date)
    run_dir.mkdir(parents=True)
    write_json(run_dir / f"{args.date}_manifest.json", manifest)
    write_json(run_dir / f"{args.date}_samples.json", samples)
    write_json(run_dir / f"{args.date}_forward_hooks.json", manifest["hooks"])
    print(f"Prepared repeat {count} samples: {run_dir}")
    print("Frozen prompts, parameters and hook context reused from:", args.reference_run)


def read_run(run_dir):
    manifests = list(run_dir.glob("*_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("run directory must contain exactly one dated manifest")
    path = manifests[0]
    return path, json.loads(path.read_text())


def http(base_url, endpoint, body=None, timeout=30):
    # Local experiment traffic must not traverse HTTP_PROXY.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base_url.rstrip("/") + endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def validate_server(info, manifest):
    # /server_info keeps CLI fields at the top level; internal_states are DP
    # runtime statistics and do not contain those fields.
    args = info.get("server_args", info)
    expected = {
        "speculative_algorithm": "EAGLE3", "speculative_num_steps": 3,
        "speculative_eagle_topk": 4, "speculative_num_draft_tokens": 16,
        "tp_size": 1, "max_running_requests": 1, "disable_cuda_graph": True,
        "disable_piecewise_cuda_graph": True,
        "disable_radix_cache": True, "enable_deterministic_inference": True,
        "context_length": manifest["context_length"], "random_seed": 0,
        "dtype": "bfloat16", "forward_hooks": manifest["hooks"],
        "model_path": manifest["target"], "speculative_draft_model_path": manifest["draft"],
    }
    errors = [f"{key}: expected {value!r}, got {args.get(key)!r}"
              for key, value in expected.items() if args.get(key) != value]
    if errors:
        raise ValueError("server configuration mismatch:\n" + "\n".join(errors))


def read_weight_context(path):
    if path is None:
        return None
    event = json.loads(path.read_text())
    if event.get("fault_kind") != "weight_bitflip" or event.get("status") not in {"injected", "restored"}:
        raise ValueError("expected a successful weight injection/restoration event")
    injected = event if event["status"] == "injected" else json.loads(
        Path(event["injected_event_path"]).read_text())
    before, after, mask = (int(injected[key], 16) for key in
                           ("before_bits_hex", "after_bits_hex", "xor_mask_hex"))
    if before ^ mask != after:
        raise ValueError("injection evidence does not satisfy bitwise XOR")
    if injected["parameter_dtype"] != "torch.bfloat16":
        raise ValueError("expected BF16 injection")
    if event["status"] == "restored":
        if (int(event["restored_bits_hex"], 16) != before or
                int(event["fault_bits_hex"], 16) != after or
                event["fault_id"] != injected["fault_id"] or
                event["worker_pid"] != injected["worker_pid"]):
            raise ValueError("restoration evidence does not match original injection")
    return {"active": event["status"] == "injected", "source_event_path": str(path.resolve()),
            "event": event, "injection_event": injected,
            "scope": "same Target worker; event records explicit inject/restore, not continuous monitoring"}


def run(args):
    manifest_path, manifest = read_run(args.run_dir)
    date = manifest["date"]
    if manifest["status"] != "prepared_not_run":
        raise ValueError("run already attempted; prepare a new directory to preserve evidence")
    samples = json.loads((args.run_dir / f"{date}_samples.json").read_text())
    server = http(args.url, "/server_info")
    validate_server(server, manifest)
    if sha256(Path(manifest["telemetry_source"])) != manifest["telemetry_source_sha256"]:
        raise ValueError("telemetry source changed since preparation; prepare again")
    for path, digest in manifest.get("telemetry_dependencies", {}).items():
        if sha256(Path(path)) != digest:
            raise ValueError("telemetry dependency changed since preparation")
    write_json(args.run_dir / f"{date}_server_info.json", server)
    context_path = Path(manifest["context_path"])
    if context_path.exists():
        raise RuntimeError("active sample marker already exists")
    weight_context = read_weight_context(args.weight_event)
    draft_point = getattr(args, "draft_fault", None)
    output_point = getattr(args, "target_output_fault", None)
    kv_fault = getattr(args, "kv_fault", False)
    if kv_fault and (output_point or draft_point or weight_context or not manifest.get("kv_fault_support")):
        raise ValueError("KV fault needs prepared support; cannot combine faults")
    kv_arm = Path(manifest["kv_fault_arm_path"]) if manifest.get("kv_fault_support") else None
    if kv_arm and kv_arm.exists():
        raise RuntimeError("stale KV fault arm file exists")
    if output_point and (draft_point or weight_context or not manifest.get("target_output_fault_support")):
        raise ValueError("Target output fault needs prepared support; cannot combine faults")
    if draft_point and (weight_context or not manifest.get("draft_fault_support")):
        raise ValueError("Draft fault needs prepared support and cannot combine with a Target weight event")
    draft_arm = Path(manifest["draft_fault_arm_path"]) if manifest.get("draft_fault_support") else None
    if draft_arm and draft_arm.exists():
        raise RuntimeError("stale Draft fault arm file exists")
    output_arm = Path(manifest["target_output_fault_arm_path"]) if manifest.get("target_output_fault_support") else None
    if output_arm and output_arm.exists():
        raise RuntimeError("stale Target output arm file exists")
    manifest["fault_injection"] = bool(kv_fault or output_point or draft_point or (weight_context and weight_context["active"]))
    if output_arm:
        manifest["target_output_fault_point"] = output_point
    manifest["fault_context"] = weight_context
    if manifest.get("draft_fault_support"):
        manifest["draft_fault_point"] = draft_point
    manifest["status"] = "running"
    manifest["start_time"] = datetime.now().isoformat()
    write_json(manifest_path, manifest)
    monitor = None
    resource_file = None
    try:
        if args.gpu is not None:
            resource_file = open(args.run_dir / f"{date}_gpu_resources.csv", "x")
            monitor = subprocess.Popen([
                "nvidia-smi", "-i", str(args.gpu),
                "--query-gpu=timestamp,index,uuid,name,memory.used,utilization.gpu,power.draw",
                "--format=csv", "-l", "2",
            ], stdout=resource_file, stderr=subprocess.STDOUT)
        for sample in samples:
            index = sample["sample_id"]
            trace = args.run_dir.resolve() / f"{date}_trace_sample_{index:03d}.jsonl"
            result = args.run_dir / f"{date}_sample_{index:03d}.json"
            if trace.exists() or result.exists():
                raise FileExistsError(f"sample {index} artifacts already exist")
            if weight_context:
                # A client-written header makes each trace's fault context self-contained.
                # Computational hook code and its source hash remain unchanged.
                header = {"kind": "fault_context", "sample_id": index,
                          "rid": sample["request"]["rid"], "context": weight_context}
                trace.write_text(json.dumps(header, allow_nan=False) + "\n")
            write_json(context_path, {"sample_id": index, "rid": sample["request"]["rid"],
                                      "trace_path": str(trace)})
            start = time.perf_counter()
            draft_event = None
            output_event = None
            kv_event = None
            try:
                if kv_fault:
                    import importlib
                    controller = importlib.import_module("fault_injection.2026-10-10_sglang_kv_bitflip")
                    kv_event = args.run_dir.resolve() / f"{date}_kv_fault_sample_{index:03d}_event.json"
                    command = {"fault_id": "kv_" + hashlib.sha256(sample["request"]["rid"].encode()).hexdigest()[:16],
                        "rid": sample["request"]["rid"], "event_path": str(kv_event),
                        "role": args.kv_role, "region": args.kv_region, "component": args.kv_component,
                        "layer": args.kv_layer, "head": args.kv_head, "dimension": args.kv_dimension,
                        "history_position": args.kv_history_position, "candidate_row": args.kv_candidate_row,
                        "bit_positions": [args.fault_bit], "occurrence": args.fault_occurrence}
                    controller.validate_command(command)
                    manifest["kv_fault_config"] = command
                    write_json(manifest_path, manifest)
                    write_json(kv_arm, command)
                if output_point:
                    import importlib
                    controller = importlib.import_module("fault_injection.2026-10-09_sglang_target_output_bitflip")
                    output_event = args.run_dir.resolve() / f"{date}_target_output_fault_sample_{index:03d}_event.json"
                    command = {"fault_id": f"{output_point}_{hashlib.sha256(sample['request']['rid'].encode()).hexdigest()[:16]}",
                        "rid": sample["request"]["rid"], "event_path": str(output_event),
                        "point": output_point, "phase": args.fault_phase, "row": args.fault_row,
                        "channel": args.fault_channel, "bit_positions": [args.fault_bit],
                        "occurrence": args.fault_occurrence}
                    controller.validate_command(command)
                    write_json(output_arm, command)
                if draft_point:
                    import importlib
                    controller = importlib.import_module("fault_injection.2026-10-09_sglang_draft_bitflip")
                    draft_event = args.run_dir.resolve() / f"{date}_draft_fault_sample_{index:03d}_event.json"
                    if draft_event.exists():
                        raise FileExistsError(draft_event)
                    command = {"fault_id": f"{draft_point}_{hashlib.sha256(sample['request']['rid'].encode()).hexdigest()[:16]}",
                               "rid": sample["request"]["rid"], "event_path": str(draft_event),
                               "point": draft_point, "phase": args.fault_phase,
                               "tap": args.fault_tap, "row": args.fault_row,
                               "channel": args.fault_channel, "bit_positions": [args.fault_bit],
                               "occurrence": args.fault_occurrence}
                    controller.validate_command(command)
                    write_json(draft_arm, command)
                response = http(args.url, "/generate", sample["request"], args.timeout)
            finally:
                context_path.unlink(missing_ok=True)
                if draft_arm:
                    draft_arm.unlink(missing_ok=True)
                if output_arm:
                    output_arm.unlink(missing_ok=True)
                if kv_arm:
                    kv_arm.unlink(missing_ok=True)
            record = {**sample, "response": response,
                      "fault_context": weight_context,
                      "client_elapsed_seconds": time.perf_counter() - start,
                      "predicted_number": answer_number(response["text"])}
            if kv_arm:
                record["kv_fault_support"] = True
            if kv_fault:
                if not kv_event.is_file():
                    raise RuntimeError("KV fault did not trigger")
                record["kv_fault_event"] = json.loads(kv_event.read_text())
            if manifest.get("kv_recording"):
                record["kv_recording"] = True
            if manifest.get("draft_fault_support"):
                record["draft_fault_support"] = True
            if output_arm:
                record["target_output_fault_support"] = True
            if output_point:
                if not output_event.is_file():
                    raise RuntimeError("Target output fault did not trigger")
                record["target_output_fault_event"] = json.loads(output_event.read_text())
            if draft_point:
                if not draft_event.is_file():
                    raise RuntimeError("Draft fault did not trigger; preserve artifacts and inspect phase/occurrence")
                record["draft_fault_event"] = json.loads(draft_event.read_text())
            record["answer_correct"] = record["predicted_number"] is not None and (
                record["predicted_number"] == record["gold_number"])
            write_json(result, record)
            errors, _ = check_sample(record, trace)
            if errors:
                raise RuntimeError(f"sample {index} recording incomplete: {'; '.join(errors)}")
            print(f"{index + 1}/{len(samples)} tokens={len(response['output_ids'])} "
                  f"correct={record['answer_correct']} trace={trace.stat().st_size} bytes",
                  flush=True)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        # Failed/timed-out requests must be stopped manually before reuse.
        context_path.unlink(missing_ok=True)
        if draft_arm:
            draft_arm.unlink(missing_ok=True)
        if output_arm:
            output_arm.unlink(missing_ok=True)
        if kv_arm:
            kv_arm.unlink(missing_ok=True)
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        if resource_file is not None:
            resource_file.close()
        manifest["resource_monitor_physical_gpu"] = args.gpu
        manifest["end_time"] = datetime.now().isoformat()
        write_json(manifest_path, manifest)
    check(args)


def check_sample(record, trace_path):
    errors = []
    response = record["response"]
    ids = response["output_ids"]
    meta = response["meta_info"]
    lp = meta.get("output_token_logprobs", [])
    if len(lp) != len(ids) or [item[1] for item in lp] != ids:
        errors.append("output token logprobs are missing or misaligned")
    if len(meta.get("output_top_logprobs", [])) != len(ids):
        errors.append("output top-8 logprobs are missing or misaligned")
    if meta.get("cached_tokens") != 0:
        errors.append("unexpected prompt cache reuse")
    events = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
    kinds = {event["kind"] for event in events}
    if record.get("target_output_fault_support"):
        states = [e for e in events if e["kind"] == "target_output_state"]
        if {e["point"] for e in states} != {"final_norm", "lm_head", "logits"}:
            errors.append("missing Target output path trace")
        output_event = record.get("target_output_fault_event")
        fired = [e for e in states if e.get("fault_applied")]
        if output_event:
            before, after, mask = (int(output_event[k], 16) for k in
                ("before_bits_hex", "after_bits_hex", "xor_mask_hex"))
            if (before ^ mask != after or output_event.get("role") != "target" or
                    output_event.get("parameters_modified") is not False or
                    not output_event.get("original_tensor_unchanged") or
                    output_event["original_sha256_before"] != output_event["original_sha256_after"] or
                    output_event.get("rid") != record["request"]["rid"] or
                    output_event.get("sample_id") != record["sample_id"] or
                    len(fired) != 1 or fired[0]["point"] != output_event["point"]):
                errors.append("invalid Target output XOR/scope/single-fire evidence")
            if any(e.get("role") == "target" and e["worker_pid"] != output_event["worker_pid"] for e in events):
                errors.append("Target output event PID differs from trace")
        elif fired:
            errors.append("unexpected Target output injection in clean run")
    if record.get("draft_fault_support"):
        points = {e.get("point") for e in events if e["kind"] == "draft_path_state"}
        if not {"interface", "fusion", "decoder_down_proj"} <= points:
            errors.append("missing Draft interface/fusion/decoder trace")
    draft_event = record.get("draft_fault_event")
    if draft_event:
        before, after, mask = (int(draft_event[key], 16) for key in
                               ("before_bits_hex", "after_bits_hex", "xor_mask_hex"))
        if before ^ mask != after or draft_event.get("role") != "draft":
            errors.append("invalid Draft bit-flip evidence")
        if (draft_event.get("rid") != record["request"]["rid"] or
                draft_event.get("sample_id") != record["sample_id"]):
            errors.append("Draft fault evidence belongs to a different sample")
        if (not draft_event.get("original_tensor_unchanged") or
                draft_event["original_sha256_before"] != draft_event["original_sha256_after"] or
                draft_event.get("target_parameters_modified") is not False):
            errors.append("Draft isolation evidence failed")
        fired = [e for e in events if e["kind"] == "draft_path_state" and e.get("fault_applied")]
        if len(fired) != 1 or fired[0]["point"] != draft_event["point"]:
            errors.append("Draft fault must fire exactly once in the requested point")
        if any(e["kind"] == "model_hidden" and e.get("role") == "draft" and
               e["worker_pid"] != draft_event["worker_pid"] for e in events):
            errors.append("Draft event PID differs from recorded model")
    if record.get("kv_recording"):
        import importlib
        kv = importlib.import_module("experiments.2026-10-09_sglang_kv_trace")
        errors.extend(kv.check_events(events))
    if record.get("kv_fault_support"):
        import importlib
        controller = importlib.import_module("fault_injection.2026-10-10_sglang_kv_bitflip")
        errors.extend(controller.check_events(record, events))
    required = {"model_hidden", "model_logits", "intermediate", "verify_block"}
    if not required <= kinds:
        errors.append(f"missing event types: {sorted(required - kinds)}")
    for role in ("target", "draft"):
        for kind in ("model_hidden", "model_logits"):
            if not any(e["kind"] == kind and e.get("role") == role for e in events):
                errors.append(f"missing {role} {kind}")
    blocks = [event for event in events if event["kind"] == "verify_block"]
    if len(blocks) != meta.get("spec_verify_ct"):
        errors.append("block trace count differs from response spec_verify_ct")
    cursor = 1  # Target prefill produces the first output token.
    for index, block in enumerate(blocks):
        if block["block_index"] != index or block["generated_index_before"] != cursor:
            errors.append("verify blocks have a gap or inconsistent output positions")
        end = block["generated_index_after"]
        if block["committed_token_ids"] != ids[cursor:end]:
            errors.append("committed block tokens differ from final output")
        row_count = block["target_probabilities"]["shape"][0]
        if any(i < 0 or i >= row_count for i in block["accepted_row_indices"]):
            errors.append("accepted row index is out of bounds")
        cursor = end
    if blocks and cursor != len(ids):
        errors.append("verify blocks do not cover all output tokens after prefill")
    for event in events:
        if event["sample_id"] != record["sample_id"] or event["rid"] != record["request"]["rid"]:
            errors.append("trace belongs to a different sample")
        if event["kind"] == "model_hidden" and event.get("role") == "target":
            context = record.get("fault_context")
            if context and event["worker_pid"] != context["event"]["worker_pid"]:
                errors.append("Target trace PID differs from weight-event PID")
            if len(event["auxiliary_hidden"]) != 3:
                errors.append("missing Target low/middle/high features")
            if event["forward_mode"] == "TARGET_VERIFY":
                h = event["final_hidden"]
                if len(h["row_indices"]) != h["original_shape"][0]:
                    errors.append("verification hidden rows were truncated")
    return sorted(set(errors)), events


def check(args):
    _, manifest = read_run(args.run_dir)
    date = manifest["date"]
    paths = sorted(args.run_dir.glob(f"{date}_sample_[0-9][0-9][0-9].json"))
    errors, records = [], []
    event_count = 0
    for path in paths:
        record = json.loads(path.read_text())
        issues, events = check_sample(record, args.run_dir / f"{date}_trace_sample_{record['sample_id']:03d}.jsonl")
        errors.extend(f"sample {record['sample_id']}: {issue}" for issue in issues)
        records.append(record)
        event_count += len(events)
    if len(records) != manifest["num_samples"]:
        errors.append(f"expected {manifest['num_samples']} results, found {len(records)}")
    metrics = [r["response"]["meta_info"] for r in records]
    proposed = sum(m.get("spec_num_proposed_drafts", 0) for m in metrics)
    correct_drafts = sum(m.get("spec_num_correct_drafts", 0) for m in metrics)
    summary = {
        "recording_checks_pass": not errors, "errors": errors,
        "samples_recorded": len(records), "trace_events": event_count,
        "correct_answers": sum(r["answer_correct"] for r in records),
        "answer_extraction_failures": sum(r["predicted_number"] is None for r in records),
        "length_truncated": sum(m["finish_reason"]["type"] == "length" for m in metrics),
        "subset_accuracy": sum(r["answer_correct"] for r in records) / len(records) if records else None,
        "total_output_tokens": sum(len(r["response"]["output_ids"]) for r in records),
        "weighted_spec_accept_rate": correct_drafts / proposed if proposed else None,
        "total_verify_blocks": sum(m.get("spec_verify_ct", 0) for m in metrics),
        "instrumented_latency_median": statistics.median(m["e2e_latency"] for m in metrics) if metrics else None,
        "performance_benchmark": False,
        "fault_injection": manifest.get("fault_injection", False),
    }
    write_json(args.run_dir / f"{date}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError("recording checks failed")


def trace_fingerprint(path):
    """Compare computation records, excluding process/request/time bookkeeping."""
    digest = hashlib.sha256()
    count = 0
    with path.open() as handle:
        for line in handle:
            event = json.loads(line)
            if event["kind"] == "fault_context":
                continue  # Provenance, not a computation; checked independently.
            for key in ("rid", "worker_pid", "timestamp_ns", "forward_index"):
                event.pop(key, None)
            if event["kind"] == "kv_cache_write":
                # Allocation addresses and artifact names are not tensor values.
                event.pop("cache_slots", None)
                event.pop("artifact_path", None)
            digest.update(json.dumps(event, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode())
            digest.update(b"\n")
            count += 1
    return digest.hexdigest(), count


def target_first_prefill(path):
    """Only pair Target prompt computations, before Draft alters candidate trees."""
    selected = {}
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("role") != "target":
            continue
        kind = event["kind"]
        if kind not in ("intermediate", "model_hidden", "model_logits") or kind in selected:
            continue
        if event.get("forward_mode") not in ("EXTEND", "MIXED", "SPLIT_PREFILL"):
            continue
        for key in ("rid", "worker_pid", "timestamp_ns", "forward_index"):
            event.pop(key, None)
        selected[kind] = event
    if set(selected) != {"intermediate", "model_hidden", "model_logits"}:
        raise ValueError("Target first prefill evidence is incomplete")
    return selected


def compare(args):
    """Read saved artifacts only; never submit requests or use a GPU."""
    _, reference = read_run(args.reference_run)
    if reference["status"] != "completed":
        raise ValueError("reference is incomplete")
    reference_summary = json.loads((args.reference_run / f"{reference['date']}_summary.json").read_text())
    if not reference_summary["recording_checks_pass"]:
        raise ValueError("reference recording is invalid")
    ref_paths = sorted(args.reference_run.glob(f"{reference['date']}_sample_[0-9][0-9][0-9].json"))
    if len(ref_paths) != reference["num_samples"]:
        raise ValueError("reference sample files are missing")
    refs = [json.loads(p.read_text()) for p in ref_paths]
    hashes = [trace_fingerprint(args.reference_run / f"{reference['date']}_trace_sample_{r['sample_id']:03d}.jsonl")
              for r in refs]
    spec_keys = ["spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_verify_ct",
                 "spec_accept_histogram", "spec_accept_length", "spec_accept_rate"]
    reports = []
    for directory in args.run_dirs:
        _, manifest = read_run(directory)
        if manifest["status"] != "completed":
            raise ValueError(f"incomplete run: {directory}")
        summary = json.loads((directory / f"{manifest['date']}_summary.json").read_text())
        if not summary["recording_checks_pass"]:
            raise ValueError(f"invalid recording: {directory}")
        config_keys = ("dataset_sha256", "num_samples", "sampling", "server_required",
                       "model_config_sha256", "target", "draft", "packages", "telemetry_source_sha256")
        if any(manifest[key] != reference[key] for key in config_keys):
            raise ValueError(f"experiment configuration differs: {directory}")
        differences = {name: [] for name in (
            "output_ids", "text", "finish_reason", "logprobs", "top8_logprobs", "spec_metrics", "internal_trace")}
        target_prefill_differences = []
        maximum_lp_diff = 0.0
        latencies = []
        for ref, fingerprint in zip(refs, hashes):
            index = ref["sample_id"]
            result = json.loads((directory / f"{manifest['date']}_sample_{index:03d}.json").read_text())
            if result["request"]["text"] != ref["request"]["text"] or result["request"]["sampling_params"] != ref["request"]["sampling_params"]:
                raise ValueError(f"request differs: sample {index} in {directory}")
            a, b = ref["response"], result["response"]
            am, bm = a["meta_info"], b["meta_info"]
            tests = {
                "output_ids": a["output_ids"] == b["output_ids"],
                "text": a["text"] == b["text"],
                "finish_reason": am["finish_reason"] == bm["finish_reason"],
                "logprobs": am["output_token_logprobs"] == bm["output_token_logprobs"],
                "top8_logprobs": am["output_top_logprobs"] == bm["output_top_logprobs"],
                "spec_metrics": all(am[key] == bm[key] for key in spec_keys),
                "internal_trace": fingerprint == trace_fingerprint(
                    directory / f"{manifest['date']}_trace_sample_{index:03d}.jsonl"),
            }
            for key, equal in tests.items():
                if not equal:
                    differences[key].append(index)
            if reference.get("draft_fault_support") and manifest.get("draft_fault_support"):
                ref_trace = args.reference_run / f"{reference['date']}_trace_sample_{index:03d}.jsonl"
                run_trace = directory / f"{manifest['date']}_trace_sample_{index:03d}.jsonl"
                if target_first_prefill(ref_trace) != target_first_prefill(run_trace):
                    target_prefill_differences.append(index)
            if tests["output_ids"]:
                maximum_lp_diff = max(maximum_lp_diff, max(
                    (abs(x[0] - y[0]) for x, y in zip(am["output_token_logprobs"], bm["output_token_logprobs"])),
                    default=0.0))
            latencies.append(bm["e2e_latency"])
        report = {
            "run_dir": str(directory), "samples_compared": len(refs),
            "all_recorded_computation_exact": not any(differences.values()),
            "different_sample_ids": differences,
            "max_logprob_diff_for_equal_token_sequences": maximum_lp_diff,
            "instrumented_latency_median": statistics.median(latencies),
            "subset_accuracy": summary["subset_accuracy"],
        }
        if reference.get("draft_fault_support") and manifest.get("draft_fault_support"):
            report["target_first_prefill_recorded_states_exact"] = not target_prefill_differences
            report["different_target_first_prefill_sample_ids"] = target_prefill_differences
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    result = {
        "reference_run": str(args.reference_run), "rounds_including_reference": len(reports) + 1,
        "all_recorded_computation_exact": all(r["all_recorded_computation_exact"] for r in reports),
        "ignored_trace_fields": ["rid", "worker_pid", "timestamp_ns", "forward_index"],
        "ignored_trace_event_kinds": ["fault_context"],
        "scope": "this dataset/config and recorded states; not a guarantee for other configurations",
        "reports": reports,
    }
    if args.output.exists():
        raise FileExistsError(f"will not overwrite {args.output}")
    write_json(args.output, result)
    print("Saved:", args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare", help="CPU only; preserve dataset prompts and prepare hook config")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reference-run", type=Path,
                   help="repeat a completed run with frozen inputs/config and shared hook context; other generation options are ignored")
    p.add_argument("--date", default=DEFAULT_DATE)
    p.add_argument("--dataset", type=Path, default=ROOT / "data_bundle/tinyGSM8k/main/test-00000-of-00001.parquet")
    p.add_argument("--target", type=Path, default=Path("/data6t/hpk/models/Qwen3-8B"))
    p.add_argument("--draft", type=Path, default=Path("/data6t/hpk/models/Qwen3-8B-eagle3"))
    p.add_argument("--num-samples", type=int)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--record-kv", action="store_true",
                   help="record actual BF16 Target/Draft KV writes, including full tensors; Qwen3 TP=1 only")
    p.add_argument("--draft-fault-support", action="store_true",
                   help="install dormant Draft-only bit-flip hooks and interface/fusion/decoder telemetry")
    p.add_argument("--target-output-fault-support", action="store_true",
                   help="install Target final norm/raw LM head/processed logits fault and trace support")
    p.add_argument("--kv-fault-support", action="store_true",
                   help="install direct KV storage faults and FA3 read/lifetime trace; requires --record-kv")
    p.set_defaults(function=prepare)
    for name in ("run", "check"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--url", default="http://127.0.0.1:30000")
        p.add_argument("--timeout", type=float, default=600)
        p.add_argument("--weight-event", type=Path,
                       help="inject/restore evidence to attach and check against Target worker PID")
        p.add_argument("--gpu", type=int, choices=(0, 2),
                       help="physical A100 to observe with nvidia-smi; does not select a compute device")
        if name == "run":
            p.add_argument("--draft-fault", choices=("interface", "fusion", "decoder_down_proj"))
            p.add_argument("--target-output-fault", choices=("final_norm", "lm_head", "logits"))
            p.add_argument("--kv-fault", action="store_true")
            p.add_argument("--kv-role", choices=("target", "draft"), default="target")
            p.add_argument("--kv-region", choices=("history", "candidate"), default="history")
            p.add_argument("--kv-component", choices=("key", "value"), default="key")
            p.add_argument("--kv-layer", type=int, default=20)
            p.add_argument("--kv-head", type=int, default=0)
            p.add_argument("--kv-dimension", type=int, default=0)
            p.add_argument("--kv-history-position", type=int, default=-1)
            p.add_argument("--kv-candidate-row", type=int, default=0)
            p.add_argument("--fault-tap", choices=("low", "middle", "high"), default="high")
            p.add_argument("--fault-row", type=int, default=-1)
            p.add_argument("--fault-channel", type=int, default=0)
            p.add_argument("--fault-bit", type=int, choices=range(32), default=14,
                           help="physical bit index: BF16 0..15; FP32 logits 0..31 (highest exponent bit 30)")
            p.add_argument("--fault-phase", choices=("prefill", "draft_decode", "draft_extend", "target_verify"), default="prefill")
            p.add_argument("--fault-occurrence", type=int, default=1)
        p.set_defaults(function=run if name == "run" else check)
    p = sub.add_parser("compare", help="CPU-only comparison of completed rounds")
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
