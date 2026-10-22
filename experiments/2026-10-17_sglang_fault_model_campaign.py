"""One owned A100 server; frozen tinyGSM8K prompts; fresh faults per request.

The two new implementation files have distinct jobs: this file prepares trials,
owns/shuts down the server, checks isolation, and compares results; the paired
fault_injection module mutates tensors and records bounded worker evidence.
Existing HTTP, answer parsing, weight control, and watchdog are reused.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from datetime import date
import hashlib
import importlib
import json
import math
import os
import random
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
client = importlib.import_module("experiments.2026-10-06_sglang_baseline")
prior = importlib.import_module("experiments.2026-10-10_sglang_kv_campaign")
weight = importlib.import_module("fault_injection.sglang_weight_bitflip")
MODULE = "fault_injection.2026-10-17_sglang_fault_models"
DATE = "2026-10-17"
GROUPS = ("compute", "compute_exp", "weight", "kv", "correlated")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    def safe(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k:safe(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)):
            return [safe(v) for v in x]
        return x
    client.write_json(temp, safe(value))
    temp.replace(path)


class Server:
    def __init__(self, args, manifest):
        self.args, self.manifest = args, manifest
        self.process = None
        self.stop_event = threading.Event()
        self.url = f"http://127.0.0.1:{args.port}"
        self.serial = 0

    def start(self):
        if self.process is not None:
            raise RuntimeError("server already running")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.args.port)) == 0:
                raise RuntimeError("port in use; will not touch existing service")
        self.serial += 1
        self.log_path = self.args.run_dir / f"{DATE}_server_{self.serial:03d}.log"
        self.handle = self.log_path.open("x")
        m = self.manifest
        cmd = ["/data6t/hpk/scripts/run_sdc_experiment.sh", "sglang", str(self.args.gpu), "--",
               "python", "-u", "-m", "sglang.launch_server", "--model-path", m["target"],
               "--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", m["draft"],
               "--speculative-num-steps", "3", "--speculative-eagle-topk", "4",
               "--speculative-num-draft-tokens", "16", "--dtype", "bfloat16",
               "--context-length", str(m["context_length"]), "--mem-fraction-static", "0.65",
               "--max-running-requests", "1", "--random-seed", "0", "--enable-deterministic-inference",
               "--disable-radix-cache", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
               "--custom-weight-loader", MODULE + ".control_loader",
               "--forward-hooks", json.dumps(m["hooks"]), "--host", "127.0.0.1",
               "--port", str(self.args.port)]
        env = dict(os.environ, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.process = subprocess.Popen(cmd, cwd=ROOT, stdout=self.handle, env=env,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        save(self.args.run_dir/f"{DATE}_server_{self.serial:03d}_process.json",
             {"pid": self.process.pid, "physical_gpu": self.args.gpu, "command": cmd})
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server startup failed: {self.log_path}")
            if "The server is fired up and ready to roll!" in self.log_path.read_text():
                info = client.http(self.url, "/server_info")
                client.validate_server(info, m)
                if info.get("server_args", info).get("custom_weight_loader") != [MODULE + ".control_loader"]:
                    raise RuntimeError("Target control loader is not registered")
                save(self.args.run_dir/f"{DATE}_server_{self.serial:03d}_info.json", info)
                return
            time.sleep(.5)
        raise TimeoutError("server warmup exceeded 300 seconds")

    def stop(self, reason="completed"):
        p = self.process
        if p is None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait(timeout=15)
        self.handle.close()
        save(self.args.run_dir/f"{DATE}_server_{self.serial:03d}_shutdown.json",
             {"pid": p.pid, "reason": reason, "returncode": p.returncode, "time": time.time()})
        self.process = None
        time.sleep(1)


def control(server, command):
    payload = weight.build_update_payload(command)
    payload["load_format"] = MODULE + ".control_loader"
    result = client.http(server.url, "/update_weights_from_tensor", payload, timeout=60)
    if not result.get("success"):
        raise RuntimeError(f"control failed: {result}")
    return json.loads(Path(command["event_path"]).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def choose_command(group, index, inventory, seed, verify_count, trial_id=None):
    if trial_id is not None and (group != "compute" or trial_id < 0):
        raise ValueError("repeated trials currently support compute only")
    event_key = [seed, group, index] if trial_id is None else [seed, group, index, trial_id]
    event_seed = int(digest(event_key)[:16], 16)
    rng = random.Random(event_seed)
    entry = rng.choices(inventory, weights=[m["numel"] for m in inventory])[0]
    # Explicit conditional sampling, not an estimate of physical fault rates.
    stratum = index if trial_id is None else index + trial_id
    bits = rng.sample(range(16), 1 + (stratum // (4 if group == "kv" else 1)) % 2)
    if verify_count < 1:
        raise ValueError("request has no verification blocks; cannot schedule this campaign's KV trial")
    command = {"kind": "compute", "module": entry["module"], "seed": event_seed,
               "phase": "prefill" if (stratum // 2) % 2 == 0 else "target_verify",
               "occurrence": 1, "pattern": "bit", "bit_positions": bits,
               "sampling": "projection weighted by parameter count; cells uniform within selected tensor"}
    if command["phase"] == "target_verify":
        command["occurrence"] = rng.randint(1, verify_count if trial_id is not None else min(8, verify_count))
    if trial_id is not None:
        command.update(trial_id=trial_id, sample_id=index,
                       time_sampling="uniform over all baseline verify blocks; prefill first call")
    if group == "compute_exp":
        command["bit_positions"] = [rng.randrange(7, 15)]
        command["stress_group"] = "exponent_only"
    elif group == "weight":
        command.update(kind="weight", phase="before_prefill", occurrence=1,
                       flat_index=rng.randrange(entry["numel"]), bit_positions=bits)
    elif group == "kv":
        command.update(kind="kv", phase="target_verify", layer=rng.randrange(36),
                       occurrence=rng.randint(1, min(8, verify_count)),
                       region="history" if index % 4 < 2 else "candidate",
                       component="key" if index % 2 == 0 else "value",
                       sampling="balanced KV region/component; uniform layer and current valid slot/head/dimension")
    elif group == "correlated":
        mode = index % 5
        command.update(pattern="word_random" if mode == 0 else
                       ("region_zero" if mode < 3 else "region_random"),
                       side=2 if mode in (1, 3) else 4,
                       geometry="logical tensor region; not a calibrated A100 warp/tile")
        command.pop("bit_positions")
    return command


def run_request(server, directory, sample, command, args, tag):
    directory.mkdir(parents=True, exist_ok=True)
    request = copy.deepcopy(sample["request"])
    request["rid"] = f"{DATE}_{tag}_{sample['sample_id']:03d}"
    request["sampling_params"]["max_new_tokens"] = args.max_new_tokens
    trace_path = directory/f"{DATE}_trace_sample_{sample['sample_id']:03d}.jsonl"
    event_path = directory/f"{DATE}_fault_sample_{sample['sample_id']:03d}.json"
    context, arm = Path(server.manifest["context_path"]), Path(server.manifest["arm_path"])
    if context.exists() or arm.exists() or trace_path.exists():
        raise RuntimeError("stale controls or duplicate request artifacts")
    c = copy.deepcopy(command)
    event = restoration = None
    failure = None
    if c:
        c.update(fault_id=request["rid"], rid=request["rid"], event_path=str(event_path))
        if c["kind"] == "weight":
            event = control(server, {"action": "inject", "fault_id": c["fault_id"],
                                    "module_name": c["module"], "flat_index": c["flat_index"],
                                    "bit_positions": c["bit_positions"], "event_path": str(event_path)})
        else:
            save(arm, c)
    save(context, {"sample_id": sample["sample_id"], "rid": request["rid"],
                   "trace_path": str(trace_path)})
    started = time.monotonic()
    try:
        response, failure = prior.guarded_request(server, request, trace_path,
                                                 event_path if c else None, 25, 180)
    finally:
        context.unlink(missing_ok=True)
        arm.unlink(missing_ok=True)
        if c and c["kind"] == "weight" and server.process is not None:
            restoration = control(server, {"action": "restore", "fault_id": c["fault_id"],
                                  "event_path": str(directory/f"{DATE}_restore_sample_{sample['sample_id']:03d}.json")})
    if c and event_path.exists():
        event = json.loads(event_path.read_text())
    if c and event is None:
        raise RuntimeError("request ended before confirmed injection")
    events, incomplete = prior.read_events(trace_path, partial=bool(failure))
    checks = []
    if response:
        if response["meta_info"].get("cached_tokens") != 0:
            checks.append("cross-request prefix cache unexpectedly used")
        verifies = [e for e in events if e["kind"] == "verify_block"]
        if len(verifies) != response["meta_info"]["spec_verify_ct"]:
            checks.append("missing verifier trace")
        committed = [t for e in verifies for t in e["committed_token_ids"]]
        # First output token comes from Target prefill, before the first verify.
        if committed != response["output_ids"][1:]:
            checks.append("committed verifier tokens do not match response")
    if c and c["kind"] != "weight":
        if len([e for e in events if e["kind"] == "fault_context"]) != 1:
            checks.append("injection did not fire exactly once")
        for cell in event["evidence"]["cells"]:
            if int(cell["before"], 16) ^ int(cell["xor"], 16) != int(cell["after"], 16):
                checks.append("invalid bit evidence")
        if c["kind"] == "kv":
            consumed = [e for e in events if e["kind"] == "kv_consumed" and
                        e["logical_visible_query_rows"] and e["bits"] == event["evidence"]["cells"][0]["after"]]
            if not consumed:
                checks.append("no logically visible FA3 consumption of modified KV")
            if c["region"] == "candidate" and response and not any(
                    e["kind"] == "kv_lifecycle" and e["action"] in {"retained_as_history", "discarded"} for e in events):
                checks.append("missing candidate lifecycle")
    if c and c["kind"] == "weight":
        before, after = int(event["before_bits_hex"],16), int(event["after_bits_hex"],16)
        if before ^ sum(1 << b for b in c["bit_positions"]) != after:
            checks.append("invalid weight bit evidence")
        if response and (not restoration or restoration["restored_bits_hex"] != event["before_bits_hex"]):
            checks.append("weight restoration not verified")
    predicted = client.answer_number(response["text"]) if response else None
    record = {"sample_id": sample["sample_id"], "question": sample["question"],
              "gold_number": sample["gold_number"], "prompt_sha256": sample["prompt_sha256"],
              "request": request, "response": response, "fault_command": c, "fault_event": event,
              "weight_restoration": restoration, "failure": failure, "trace_path": str(trace_path),
              "predicted_number": predicted, "answer_correct": predicted is not None and predicted == sample["gold_number"],
              "checks": checks, "checks_pass": not checks, "incomplete_lines": incomplete,
              "trace_events": len(events), "elapsed_seconds": time.monotonic()-started}
    if c and "trial_id" in c:
        record["trial_id"] = c["trial_id"]
    save(directory/f"{DATE}_sample_{sample['sample_id']:03d}.json", record)
    if checks:
        raise RuntimeError(f"recording/injection validation failed: {checks}")
    return record


def compare_record(reference, trial):
    if not trial["response"]:
        return {"request_failure": trial["failure"], "baseline_correct_to_failure": reference["answer_correct"]}
    a, b = reference["response"], trial["response"]
    aid, bid = a["output_ids"], b["output_ids"]
    divergence = next((i for i,(x,y) in enumerate(zip(aid,bid)) if x!=y),
                      min(len(aid),len(bid)) if len(aid)!=len(bid) else None)
    def hidden(path, fault=None):
        # Pair only the same role/mode/token positions; exclude post-divergence interpretation.
        output = {}
        diagnostics = {"target_max_rms": 0.0, "target_max_abs_finite": 0.0,
                       "target_taps_nonfinite": 0, "draft_hidden_nonfinite": 0,
                       "target_logits_nonfinite": 0, "fault_row_accepted": None}
        verify_ordinal = 0
        for line in Path(path).open():
            e = json.loads(line)
            if e["kind"] == "model_hidden" and e["role"] == "draft":
                diagnostics["draft_hidden_nonfinite"] += sum(e["hidden"]["nonfinite_count"])
            if e["kind"] == "verify_block":
                verify_ordinal += 1
                diagnostics["target_logits_nonfinite"] += sum(e["probabilities"]["nonfinite_count"])
                if fault and fault.get("phase") == "target_verify" and verify_ordinal == fault["occurrence"]:
                    row = fault["evidence"]["cells"][0]["row"]
                    diagnostics["fault_row_accepted"] = row in e["accepted_rows"]
            if e["kind"] == "model_hidden" and e["role"] == "target":
                h = e["hidden"]
                output[(e["forward_mode"],e["forward_index"])] = h
                diagnostics["target_max_rms"] = max(diagnostics["target_max_rms"], max(h["rms_full"], default=0))
                diagnostics["target_max_abs_finite"] = max(diagnostics["target_max_abs_finite"], max(h["max_abs_finite"], default=0))
                diagnostics["target_taps_nonfinite"] += sum(sum(t["nonfinite_count"]) for t in e["taps"])
        return output, diagnostics
    ah, baseline_diagnostics = hidden(reference["trace_path"])
    bh, trial_diagnostics = hidden(trial["trace_path"], trial.get("fault_event"))
    differences = []
    nonfinite = 0
    for k, h in bh.items():
        nonfinite += sum(h["nonfinite_count"])
        if k in ah and ah[k]["positions"] == h["positions"] and ah[k]["slice_values"] != h["slice_values"]:
            differences.append(k[1])
    al = [x[0] for x in a["meta_info"]["output_token_logprobs"]]
    bl = [x[0] for x in b["meta_info"]["output_token_logprobs"]]
    prefix = min(divergence if divergence is not None else len(aid),len(al),len(bl))
    return {"output_ids_equal": aid==bid, "first_divergent_token": divergence,
            "baseline_correct_to_wrong": reference["answer_correct"] and not trial["answer_correct"],
            "output_changed_task_correct": aid!=bid and trial["answer_correct"],
            "baseline_output_tokens":len(aid),"trial_output_tokens":len(bid),
            "baseline_accept_rate":a["meta_info"]["spec_accept_rate"],
            "trial_accept_rate":b["meta_info"]["spec_accept_rate"],
            "same_token_prefix_max_logprob_diff":max((abs(x-y) for x,y in zip(al[:prefix],bl[:prefix])
                if isinstance(x,(float,int)) and isinstance(y,(float,int)) and math.isfinite(x) and math.isfinite(y)),default=None),
            "aligned_target_forwards_hidden_slice_changed":len(differences),
            "first_changed_aligned_target_forward":min(differences,default=None),
            "target_recorded_rows_nonfinite_elements":nonfinite,
            "baseline_trace_diagnostics":baseline_diagnostics,"trial_trace_diagnostics":trial_diagnostics,
            "hidden_alignment_limit":"same forward/mode/positions; after tree/path changes not a causal distance"}


def summary(records):
    pairs = [r["comparison"] for r in records if "comparison" in r]
    successful = [r for r in records if r["response"]]
    meta = [r["response"]["meta_info"] for r in successful]
    return {"requests":len(records),"completed":len(successful),"checks_pass":all(r["checks_pass"] for r in records),
            "correct_answers":sum(r["answer_correct"] for r in successful),
            "accuracy_completed":sum(r["answer_correct"] for r in successful)/len(successful) if successful else None,
            "request_failures":len(records)-len(successful),
            "changed_outputs":sum(p.get("output_ids_equal") is False for p in pairs),
            "correct_to_wrong":sum(p.get("baseline_correct_to_wrong",False) for p in pairs),
            "changed_but_correct":sum(p.get("output_changed_task_correct",False) for p in pairs),
            "output_tokens":sum(m["completion_tokens"] for m in meta),
            "length_truncated":sum(m["finish_reason"]["type"]=="length" for m in meta),
            "weighted_accept_rate":sum(m["spec_num_correct_drafts"] for m in meta)/sum(m["spec_num_proposed_drafts"] for m in meta) if meta else None,
            "instrumented_not_performance_benchmark":True,"not_an_online_detection_evaluation":True}


def compute_trial_row(record, reference):
    """Small, auditable rows for position and within-input comparisons; no tensors."""
    c, event, pair = record["fault_command"], record.get("fault_event") or {}, record["comparison"]
    evidence = event.get("evidence", {})
    cell = next(iter(evidence.get("cells", [])), {})
    def bf16(key):
        return struct.unpack(">f", (int(cell[key], 16) << 16).to_bytes(4, "big"))[0] if key in cell else None
    before, after = bf16("before"), bf16("after")
    bit_positions = c["bit_positions"]
    bit_class = ("exponent_involved" if any(7 <= b <= 14 for b in bit_positions) else
                 "sign_only" if bit_positions == [15] else
                 "sign_and_mantissa" if 15 in bit_positions else "mantissa_only")
    responded = record["response"] is not None
    changed = pair.get("output_ids_equal") is False
    harmful = responded and reference["answer_correct"] and not record["answer_correct"]
    outcome = ("request_failure" if not responded else "correct_to_wrong" if harmful else
               "wrong_to_correct" if not reference["answer_correct"] and record["answer_correct"] else
               "output_unchanged" if not changed else "changed_but_correct" if record["answer_correct"] else
               "changed_still_wrong")
    layer = int(c["module"].split(".layers.")[1].split(".")[0])
    diagnostics = pair.get("trial_trace_diagnostics", {})
    return {"sample_id": record["sample_id"], "trial_id": c["trial_id"], "seed": c["seed"],
            "module": c["module"], "operator": c["module"].rsplit(".", 1)[1], "layer": layer,
            "layer_band": f"{layer//12*12:02d}_{layer//12*12+11:02d}",
            "phase": c["phase"], "occurrence": c["occurrence"], "bit_count": len(bit_positions),
            "bit_positions": bit_positions, "bit_class": bit_class,
            "row": cell.get("row"), "column": cell.get("column"),
            "coordinate": event.get("output_coordinate"), "before_bits": cell.get("before"),
            "after_bits": cell.get("after"),
            "before_value": before if before is not None and math.isfinite(before) else None,
            "after_value": after if after is not None and math.isfinite(after) else None,
            "after_class": "missing" if after is None else "nan" if math.isnan(after) else
                           "inf" if math.isinf(after) else "finite",
            "baseline_correct": reference["answer_correct"], "completed": responded,
            "checks_pass": record["checks_pass"], "outcome": outcome,
            "output_changed": changed, "correct_to_wrong": harmful,
            "first_divergent_token": pair.get("first_divergent_token"),
            "output_tokens": pair.get("trial_output_tokens"),
            "accept_rate_delta": pair["trial_accept_rate"]-pair["baseline_accept_rate"] if responded else None,
            "logprob_prefix_max_diff": pair.get("same_token_prefix_max_logprob_diff"),
            "aligned_hidden_slice_changes": pair.get("aligned_target_forwards_hidden_slice_changed"),
            "target_hidden_nonfinite": pair.get("target_recorded_rows_nonfinite_elements"),
            "fault_row_accepted": diagnostics.get("fault_row_accepted"),
            "trace_diagnostics": diagnostics, "event_path": c["event_path"],
            "sample_record": str(Path(record["trace_path"]).with_name(f"{DATE}_sample_{record['sample_id']:03d}.json"))}


def position_analysis(rows):
    def aggregate(items):
        good = [r for r in items if r["completed"]]
        eligible = [r for r in good if r["baseline_correct"]]
        harmful = sum(r["correct_to_wrong"] for r in eligible)
        return {"trials": len(items), "completed": len(good), "failures": len(items)-len(good),
                "output_changed": sum(r["output_changed"] for r in good),
                "baseline_correct_completed_trials": len(eligible), "correct_to_wrong": harmful,
                "harmful_rate_given_baseline_correct": harmful/len(eligible) if eligible else None,
                "target_hidden_nonfinite_trials": sum(bool(r["target_hidden_nonfinite"]) for r in good),
                "finite_injection_harmful": sum(r["correct_to_wrong"] and r["after_class"] == "finite" for r in good),
                "outcomes": dict(Counter(r["outcome"] for r in items))}
    by_input = []
    inputs = defaultdict(list)
    for row in rows:
        inputs[row["sample_id"]].append(row)
    for sid, items in sorted(inputs.items()):
        deltas = [r["accept_rate_delta"] for r in items if r["completed"]]
        by_input.append({"sample_id": sid, "baseline_correct": items[0]["baseline_correct"], **aggregate(items),
                         "accept_rate_delta_min": min(deltas, default=None),
                         "accept_rate_delta_median": statistics.median(deltas) if deltas else None,
                         "accept_rate_delta_max": max(deltas, default=None),
                         "mixed_output_change_outcomes": len({r["output_changed"] for r in items if r["completed"]}) > 1,
                         "mixed_harmful_outcomes": len({r["correct_to_wrong"] for r in items if r["completed"]}) > 1})
    strata = {}
    dimensions = {
        "phase_bit_count": lambda r: f"{r['phase']}__{r['bit_count']}bit",
        "operator": lambda r: r["operator"], "layer": lambda r: str(r["layer"]),
        "layer_band": lambda r: r["layer_band"], "bit_class": lambda r: r["bit_class"],
        "bit_mask": lambda r: ",".join(map(str, sorted(r["bit_positions"]))),
        "injected_value_class": lambda r: r["after_class"],
        "operator_component": lambda r: r["operator"]+"__"+(r.get("coordinate") or {}).get("projection_component", "unknown"),
        "phase_operator": lambda r: r["phase"]+"__"+r["operator"],
        "fault_row_accepted": lambda r: str(r["fault_row_accepted"]),
    }
    for dimension, key in dimensions.items():
        buckets = defaultdict(list)
        for row in rows:
            buckets[key(row)].append(row)
        strata[dimension] = {name: aggregate(items) for name, items in sorted(buckets.items())}
    return {"overall": aggregate(rows), "by_input": by_input, "strata": strata,
            "inputs_with_mixed_output_change_outcomes": sum(x["mixed_output_change_outcomes"] for x in by_input),
            "inputs_with_mixed_harmful_outcomes": sum(x["mixed_harmful_outcomes"] for x in by_input),
            "limits": ["Descriptive conditional fault outcomes, not physical occurrence rates or online detection rates.",
                       "Layer/operator/bit comparisons include different sampled cells and inputs; no isolated causal ranking.",
                       "Repeated trials within an input share that input; do not treat them as independently sampled prompts.",
                       "Rejected rows can still influence accepted attention results; fault_row_accepted is not a masking proof.",
                       "Hidden slice alignment by forward and positions alone is not causal alignment after tree divergence."]}


def write_position_report(run_dir, rows):
    analysis = position_analysis(rows)
    save(run_dir/f"{DATE}_position_analysis.json", analysis)
    overall = analysis["overall"]
    text = ["# Target 单次计算故障：重复注入位置分析", "",
            f"完成 {overall['completed']}/{overall['trials']} 次故障请求；请求失败 {overall['failures']}。",
            f"输出序列改变 {overall['output_changed']} 次；正常答对变错 {overall['correct_to_wrong']} 次，"
            f"分母为 {overall['baseline_correct_completed_trials']} 次正常答对且故障请求完成的试验。", "",
            f"同一输入内，输出是否变化存在差异的题目：{analysis['inputs_with_mixed_output_change_outcomes']}；"
            f"是否由对变错存在差异的题目：{analysis['inputs_with_mixed_harmful_outcomes']}。", "",
            "下表只描述实际抽样结果，各子组样本量不同，不直接证明哪个位置更危险。", ""]
    for dimension in ("phase_bit_count", "operator", "layer_band", "bit_class", "injected_value_class", "operator_component"):
        text += [f"## {dimension}", "", "| 条件 | 请求数 | 完成 | 输出变化 | 正常答对变错 / 可判断分母 |", "|---|---:|---:|---:|---:|"]
        for name, item in analysis["strata"][dimension].items():
            text.append(f"| {name} | {item['trials']} | {item['completed']} | {item['output_changed']} | "
                        f"{item['correct_to_wrong']} / {item['baseline_correct_completed_trials']} |")
        text.append("")
    text += ["## 边界", ""] + [f"- {x}" for x in analysis["limits"]]
    (run_dir/f"{DATE}_position_analysis.md").write_text("\n".join(text)+"\n")


def main():
    global DATE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--reference-run",type=Path,default=ROOT/"results/2026-10-07_qwen3_eagle3_baseline_full")
    p.add_argument("--gpu",type=int,choices=(0,2),default=2)
    p.add_argument("--port",type=int,default=30002)
    p.add_argument("--num-samples",type=int,default=100)
    p.add_argument("--max-new-tokens",type=int,default=1024)
    p.add_argument("--seed",type=int,default=20261017)
    p.add_argument("--groups",nargs="+",choices=GROUPS,default=list(GROUPS))
    p.add_argument("--trials-per-sample",type=int,default=1,
                   help="Repeated compute-only events per input; multiples of four balance phase and bit count")
    p.add_argument("--date",default=DATE,help="YYYY-MM-DD prefix for newly created artifacts")
    args = p.parse_args()
    DATE = date.fromisoformat(args.date).isoformat()
    if args.trials_per_sample < 1 or (args.trials_per_sample > 1 and
            (args.groups != ["compute"] or args.trials_per_sample % 4)):
        p.error("repeated trials require --groups compute and a positive multiple of four trials")
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists():
        raise FileExistsError("will not overwrite an existing campaign")
    _, reference = client.read_run(args.reference_run)
    if client.sha256(Path(reference["dataset"])) != reference["dataset_sha256"]:
        raise RuntimeError("reference dataset changed")
    samples = json.loads((args.reference_run/f"{reference['date']}_samples.json").read_text())
    if not 1 <= args.num_samples <= len(samples):
        raise ValueError("invalid sample count")
    samples = samples[:args.num_samples]
    args.run_dir.mkdir(parents=True)
    manifest = {k:reference[k] for k in ("target","draft","context_length","dataset","dataset_sha256","model_config_sha256")}
    manifest.update(context_path=str(args.run_dir/f"{DATE}_active_sample.json"),
                    arm_path=str(args.run_dir/f"{DATE}_fault_arm.json"),
                    seed=args.seed,num_samples=len(samples),max_new_tokens=args.max_new_tokens,
                    groups=args.groups, trials_per_sample=args.trials_per_sample, date=DATE,
                    expected_fault_requests=len(samples)*args.trials_per_sample*len(args.groups),
                    verify_time_sampling="all baseline blocks" if args.trials_per_sample > 1 else "first 8 baseline blocks",
                    status="starting", raw_tensor_dump=False)
    manifest["hooks"]=[{"name":"request_fault_models_and_bounded_trace","target_modules":["model"],
                        "hook_factory":MODULE+":create_fault_model_hook",
                        "config":{"context_path":manifest["context_path"],"arm_path":manifest["arm_path"]}},
                       {"name":"bounded_draft_probabilities","target_modules":["logits_processor"],
                        "hook_factory":MODULE+":create_draft_probability_hook",
                        "config":{"context_path":manifest["context_path"]}}]
    manifest["source_sha256"]={str(f.relative_to(ROOT)):client.sha256(f) for f in
        [Path(__file__).resolve(), ROOT/(MODULE.replace('.','/')+'.py'),
         ROOT/'fault_injection/sglang_weight_bitflip.py',
         ROOT/'experiments/2026-10-06_sglang_baseline_trace.py',
         ROOT/'experiments/2026-10-06_sglang_baseline.py',
         ROOT/'experiments/2026-10-10_sglang_kv_campaign.py',
         ROOT/'fault_injection/2026-10-10_sglang_kv_bitflip.py']}
    save(args.run_dir/f"{DATE}_manifest.json",manifest)
    save(args.run_dir/f"{DATE}_samples.json",samples)
    server = Server(args,manifest)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    all_summary = {}
    try:
        server.start()
        manifest['status']='running'
        save(args.run_dir/f"{DATE}_manifest.json",manifest)
        inventory=control(server,{"action":"inventory","event_path":str(args.run_dir/f"{DATE}_weight_inventory.json")})["modules"]
        if len(inventory)!=144 or any(m['dtype']!='torch.bfloat16' for m in inventory):
            raise RuntimeError("Target projection inventory differs from expected Qwen3 BF16 layout")
        normal=[]
        for sample in samples:
            record=run_request(server,args.run_dir/f"{DATE}_baseline",sample,None,args,"baseline")
            if not record['response']:
                raise RuntimeError("normal request failed")
            if args.max_new_tokens==reference['max_new_tokens']:
                old=json.loads((args.reference_run/f"{reference['date']}_sample_{sample['sample_id']:03d}.json").read_text())
                if not prior.response_equal(old['response'],record['response']):
                    raise RuntimeError("fresh baseline differs from frozen prior output/logprobs/spec metrics")
            normal.append(record)
            print(f"baseline {len(normal)}/{len(samples)}",flush=True)
        all_summary['baseline']=summary(normal)
        save(args.run_dir/f"{DATE}_summary.json",all_summary)
        plan=[]
        for group in args.groups:
            for s,n in zip(samples,normal):
                for trial_id in range(args.trials_per_sample):
                    c=choose_command(group,s['sample_id'],inventory,args.seed,n['response']['meta_info']['spec_verify_ct'],
                                     trial_id if args.trials_per_sample > 1 else None)
                    plan.append({'group':group,'sample_id':s['sample_id'],'trial_id':trial_id,'command':c})
        save(args.run_dir/f"{DATE}_fault_manifest.json",plan)
        samples_by_id = {s['sample_id']: s for s in samples}
        normal_by_id = {r['sample_id']: r for r in normal}
        position_rows = []
        for group in args.groups:
            records=[]
            group_plan = [entry for entry in plan if entry['group'] == group]
            for entry in group_plan:
                sample = samples_by_id[entry['sample_id']]
                reference_record = normal_by_id[entry['sample_id']]
                if server.process is None:
                    server.start()
                directory = args.run_dir/f"{DATE}_{group}"
                tag = group
                if args.trials_per_sample > 1:
                    directory /= f"{DATE}_trial_{entry['trial_id']:03d}"
                    tag += f"_trial_{entry['trial_id']:03d}"
                record=run_request(server,directory,sample,entry['command'],args,tag)
                record['comparison']=compare_record(reference_record,record)
                save(directory/f"{DATE}_sample_{sample['sample_id']:03d}.json",record)
                records.append(record)
                if args.trials_per_sample > 1:
                    row = compute_trial_row(record, reference_record)
                    position_rows.append(row)
                    with (args.run_dir/f"{DATE}_trial_table.jsonl").open('a') as handle:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                all_summary[group]=summary(records)
                all_summary[group]['expected_requests'] = len(group_plan)
                # Preserve strata; mixed group totals describe this campaign, not a natural fault distribution.
                strata={}
                for r in records:
                    c=r['fault_command'];key=json.dumps([c['pattern'],len(c.get('bit_positions',[])),c.get('region'),c.get('component'),c['phase'],c.get('side')])
                    strata.setdefault(key,[]).append(r)
                all_summary[group]['strata']={k:summary(v) for k,v in strata.items()}
                save(args.run_dir/f"{DATE}_summary.json",all_summary)
                print(f"{group} {len(records)}/{len(group_plan)} sample={sample['sample_id']} trial={entry['trial_id']} "
                      f"correct={all_summary[group]['correct_answers']} changed={all_summary[group]['changed_outputs']}",flush=True)
                if len(records)%10==0 or len(records)==len(group_plan):
                    if server.process is None:
                        server.start()
                    probe=run_request(server,args.run_dir/f"{DATE}_health_{group}_{len(records):03d}",samples[0],None,args,f"health_{group}_{len(records)}")
                    if not probe['response'] or not prior.response_equal(normal[0]['response'],probe['response']):
                        raise RuntimeError("post-fault health probe differs from normal baseline")
                    manifest.update(completed_fault_requests=sum(x['requests'] for k,x in all_summary.items() if k != 'baseline'),
                                    last_sample_id=sample['sample_id'], last_trial_id=entry['trial_id'])
                    save(args.run_dir/f"{DATE}_manifest.json",manifest)
                if args.trials_per_sample > 1 and len(records)%args.trials_per_sample == 0:
                    write_position_report(args.run_dir, position_rows)
            if args.trials_per_sample > 1:
                write_position_report(args.run_dir, position_rows)
                counts = Counter(r['sample_id'] for r in records)
                if len(counts) != len(samples) or set(counts.values()) != {args.trials_per_sample}:
                    raise RuntimeError("per-input trial counts differ from the requested design")
        manifest['status']='completed'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        raise
    finally:
        server.stop(manifest['status'])
        Path(manifest['context_path']).unlink(missing_ok=True)
        Path(manifest['arm_path']).unlink(missing_ok=True)
        save(args.run_dir/f"{DATE}_manifest.json",manifest)


if __name__=='__main__':
    main()
