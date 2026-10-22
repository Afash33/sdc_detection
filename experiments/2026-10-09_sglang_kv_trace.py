"""Read-only KV write telemetry and CPU comparison for the Qwen3 EAGLE3 experiment.

Loaded through --forward-hooks. Reuses the frozen 2026-10-06 recorder without
editing it. Records actual BF16 cache rows after writes, including speculative
rows; those rows are not automatically committed history. TP=1, graphs off.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
from pathlib import Path
from types import MethodType

import torch

base = importlib.import_module("experiments.2026-10-06_sglang_baseline_trace")


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_summary(value):
    x = value.double().flatten(1)
    finite = torch.isfinite(x)
    safe = torch.where(finite, x, 0.)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": tensor_hash(value),
            "l2_per_row": torch.linalg.vector_norm(safe, dim=1).tolist(),
            "max_abs_finite_per_row": safe.abs().amax(dim=1).tolist(),
            "nonfinite_count_per_row": (~finite).sum(dim=1).tolist()}


def read_written_rows(pool, layer_id, locations):
    """Read only the just-written slots, not unused cache capacity."""
    if getattr(pool, "layer_transfer_counter", None) is not None:
        raise RuntimeError("KV recorder does not support cache offloading")
    if locations.ndim != 1 or locations.numel() == 0:
        raise ValueError("expected nonempty 1D cache locations")
    k = pool._get_key_buffer(layer_id)
    v = pool._get_value_buffer(layer_id)
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("KV experiment requires BF16 cache storage")
    if bool((locations <= 0).any()) or bool((locations >= k.shape[0]).any()):
        raise ValueError("cache locations include dummy or invalid slots")
    # Diagnostic eager run: synchronizing also covers alternate cache-write streams.
    if k.is_cuda:
        torch.cuda.synchronize(k.device)
    return k[locations.long()].detach().cpu(), v[locations.long()].detach().cpu()


def _install_pool(pool):
    if getattr(pool, "_sdc_kv_recorder_installed", False):
        return
    original = pool.set_kv_buffer

    @functools.wraps(original)
    def write(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        result = original(layer, loc, cache_k, cache_v, *args, **kwargs)
        active = getattr(self, "_sdc_kv_active", None)
        if active is None:
            return result
        override = kwargs.get("layer_id_override", args[2] if len(args) > 2 else None)
        layer_id = layer.layer_id if override is None else override
        locations = loc.detach()
        if len(active["positions"]) != locations.numel():
            raise RuntimeError("KV rows and model positions differ; refusing ambiguous alignment")
        k, v = read_written_rows(self, layer_id, locations)
        event = {
            "kind": "kv_cache_write", "role": active["role"],
            "layer_id": int(layer_id), "forward_index": active["index"],
            "forward_mode": active["mode"], "positions": active["positions"],
            "input_token_ids": active["tokens"], "cache_slots": loc.cpu().tolist(),
            "row_semantics": "write-time rows; speculative rows may later be rejected or relocated",
            "artifact_path": str(active["artifact"]),
            "key": tensor_summary(k), "value": tensor_summary(v),
        }
        active["captures"].append({"layer_id": int(layer_id), "key": k, "value": v})
        base._write(active["context"], event)
        return result

    pool.set_kv_buffer = MethodType(write, pool)
    pool._sdc_kv_recorder_installed = True


def create_model_trace_hook(config):
    normal_hook = base.create_model_trace_hook(config)
    installed = False
    state = {}
    current_rid = None
    index = 0

    def before(module, args):
        nonlocal current_rid, index
        batch = args[2]
        pool = batch.token_to_kv_pool
        _install_pool(pool)
        context = base._context(config)
        pool._sdc_kv_active = None
        state.clear()
        if context is None or batch.forward_mode.is_idle():
            return
        if current_rid != context["rid"]:
            current_rid, index = context["rid"], 0
        role = "draft" if int(module.config.num_hidden_layers) == 1 else "target"
        trace = Path(context["trace_path"])
        artifact = trace.with_name(trace.stem.replace("trace_sample", "kv_sample")
                                   + f"_{role}_forward_{index:03d}.pt")
        state.update(context=context, role=role, index=index,
                     mode=batch.forward_mode.name,
                     positions=base._tolist(args[1]), tokens=base._tolist(args[0]),
                     artifact=artifact, captures=[])
        pool._sdc_kv_active = state
        index += 1

    def after(module, args, output):
        nonlocal installed
        normal_hook(module, args, output)
        if not installed:
            module.register_forward_pre_hook(before)
            installed = True
        if state:
            if not state["captures"]:
                raise RuntimeError("no KV writes intercepted; this backend is unsupported")
            if state["artifact"].exists():
                raise FileExistsError(state["artifact"])
            with state["artifact"].open("xb") as handle:
                torch.save({"role": state["role"], "forward_index": state["index"],
                            "forward_mode": state["mode"], "positions": state["positions"],
                            "input_token_ids": state["tokens"],
                            "captures": state["captures"]}, handle)
            args[2].token_to_kv_pool._sdc_kv_active = None
            state.clear()
        return None

    return after


create_logits_trace_hook = base.create_logits_trace_hook


def check_events(events):
    """Verify tensor artifacts, complete layers and write metadata on CPU."""
    errors = []
    grouped = {}
    for e in events:
        if e["kind"] == "kv_cache_write":
            grouped.setdefault(e["artifact_path"], []).append(e)
    if not grouped:
        return ["no KV cache writes recorded"]
    roles = set()
    for path, writes in grouped.items():
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        roles.add(artifact["role"])
        expected = set(range(36)) if artifact["role"] == "target" else {0}
        if {e["layer_id"] for e in writes} != expected or len(writes) != len(expected):
            errors.append("incomplete or duplicated KV layers")
        tensors = {c["layer_id"]: c for c in artifact["captures"]}
        for e in writes:
            if e["positions"] != artifact["positions"] or e["input_token_ids"] != artifact["input_token_ids"]:
                errors.append("KV positions/tokens differ from artifact")
            for name in ("key", "value"):
                value = tensors[e["layer_id"]][name]
                if tensor_hash(value) != e[name]["sha256"]:
                    errors.append("KV artifact hash differs from recorded cache")
                if value.shape[0] != len(e["positions"]) or len(e["cache_slots"]) != value.shape[0]:
                    errors.append("KV tensor rows are not aligned")
    if roles != {"target", "draft"}:
        errors.append("missing Target or Draft KV")
    return sorted(set(errors))


def aligned_difference(a, b):
    """Compare corresponding cache rows; report nonfinite differences separately."""
    if a.shape != b.shape:
        raise ValueError("cannot compare different KV tensor shapes")
    bit_equal = a.view(torch.int16) == b.view(torch.int16)
    x, y = a.double(), b.double()
    finite = torch.isfinite(x) & torch.isfinite(y)
    difference = torch.where(finite, x - y, 0.)
    return {"rows": a.shape[0], "bitwise_equal": bool(bit_equal.all()),
            "rows_bitwise_equal": int(bit_equal.flatten(1).all(dim=1).sum()),
            "max_abs_difference_finite": float(difference.abs().max()),
            "l2_difference_finite": float(torch.linalg.vector_norm(difference)),
            "reference_nonfinite_elements": int((~torch.isfinite(x)).sum()),
            "trial_nonfinite_elements": int((~torch.isfinite(y)).sum())}


def compare_runs(reference, trials, output):
    """First-prefill comparison uses identical causal input prefixes only."""
    def load_initial(directory, role):
        paths = list(directory.glob(f"*_kv_sample_000_{role}_forward_000.pt"))
        if len(paths) != 1:
            raise ValueError(f"expected one initial {role} KV artifact in {directory}")
        return torch.load(paths[0], map_location="cpu", weights_only=True)

    reports = []
    for directory in trials:
        roles = {}
        for role in ("target", "draft"):
            a, b = load_initial(reference, role), load_initial(directory, role)
            if len(a["positions"]) != len(b["positions"]):
                raise ValueError("initial prefills have different lengths")
            common = 0
            for ap, bp, at, bt in zip(a["positions"], b["positions"], a["input_token_ids"], b["input_token_ids"]):
                if ap != bp or at != bt:
                    break
                common += 1
            if common == 0:
                raise ValueError("no common causal input prefix")
            ac, bc = ({c["layer_id"]: c for c in x["captures"]} for x in (a, b))
            roles[role] = {"compared_prefix_rows": common,
                           "excluded_suffix_rows": len(a["positions"]) - common,
                           "layers": {str(i): {name: aligned_difference(ac[i][name][:common], bc[i][name][:common])
                                                for name in ("key", "value")} for i in sorted(ac)}}
        reports.append({"run_dir": str(directory), "first_prefill": roles})
    result = {"reference_run": str(reference), "scope": "same-input first prefills only; post-divergence candidate rows are saved, not paired",
              "reports": reports}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Saved KV comparison:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only comparison of recorded first-prefill KV")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare_runs(args.reference_run, args.run_dirs, args.output)
