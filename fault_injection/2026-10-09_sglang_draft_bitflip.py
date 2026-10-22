"""Request-scoped BF16 activation faults on the Draft-only EAGLE3 path.

SGLang attaches the installer to nested `model` in both workers. It installs
sub-hooks only on the one-layer EAGLE3 Draft during unarmed server warmup.
Interface faults replace fc input with a private clone, preserving Target data.
No parameters, embeddings, LM heads, tree or verifier code are modified.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from pathlib import Path

import torch

from fault_injection.sglang_activation_bitflip import (
    _atomic_write_json, _parse_bit_positions, _signed_int16,
)

base = importlib.import_module("experiments.2026-10-06_sglang_baseline_trace")
POINTS = ("interface", "fusion", "decoder_down_proj")
MODULES = {"interface": "model.fc:input", "fusion": "model.fc:output",
           "decoder_down_proj": "model.layers.0.mlp.down_proj:output"}
PHASES = {"prefill": {"EXTEND", "MIXED", "SPLIT_PREFILL"},
          "draft_decode": {"DECODE"},
          "draft_extend": {"DRAFT_EXTEND", "DRAFT_EXTEND_V2"}}


def _digest(value):
    data = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _number(value):
    if math.isnan(value):
        return None, "nan"
    if math.isinf(value):
        return None, "positive_inf" if value > 0 else "negative_inf"
    return value, "finite"


def validate_command(command):
    if command.get("point") not in POINTS or command.get("phase") not in PHASES:
        raise ValueError("unsupported Draft injection point or phase")
    if command["point"] in ("interface", "fusion") and command["phase"] == "draft_decode":
        raise ValueError("Draft decode bypasses the three-feature fc interface")
    for name in ("fault_id", "rid", "event_path"):
        if not command.get(name):
            raise ValueError(f"{name} is required")
    _parse_bit_positions(command.get("bit_positions"))
    if command.get("tap", "high") not in ("low", "middle", "high"):
        raise ValueError("tap must be low, middle or high")
    if int(command.get("occurrence", 1)) < 1:
        raise ValueError("occurrence must be positive")
    if int(command.get("channel", 0)) < 0:
        raise ValueError("channel must be nonnegative")


def copy_and_flip(tensor, command, hidden_size):
    """Modify one cloned element; verify the entire original remains bit-identical."""
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2 or not tensor.numel():
        raise TypeError("Draft fault requires a nonempty [rows, channels] BF16 tensor")
    before_hash = _digest(tensor)
    modified = tensor.detach().clone(memory_format=torch.contiguous_format)
    row = int(command.get("row", -1))
    row = row + tensor.shape[0] if row < 0 else row
    channel = int(command.get("channel", 0))
    if command["point"] == "interface":
        if tensor.shape[1] != 3 * hidden_size or channel >= hidden_size:
            raise ValueError("interface expects three Target feature groups")
        channel += ("low", "middle", "high").index(command.get("tap", "high")) * hidden_size
    if not 0 <= row < tensor.shape[0] or not 0 <= channel < tensor.shape[1]:
        raise IndexError("Draft fault row/channel is out of bounds")
    bits = modified.view(torch.int16)
    before = int(bits[row, channel].item()) & 0xffff
    before_value, before_class = _number(float(modified[row, channel].item()))
    mask = sum(1 << b for b in _parse_bit_positions(command["bit_positions"]))
    with torch.no_grad():
        bits[row, channel] = _signed_int16(before ^ mask)
    after = int(bits[row, channel].item()) & 0xffff
    after_value, after_class = _number(float(modified[row, channel].item()))
    after_hash = _digest(tensor)
    if after != before ^ mask or before_hash != after_hash:
        raise RuntimeError("bit verification or original-tensor isolation failed")
    return modified, {
        "tensor_shape": list(tensor.shape), "tensor_dtype": str(tensor.dtype),
        "row": row, "channel": channel, "flat_index": row * tensor.shape[1] + channel,
        "tap": command.get("tap", "high") if command["point"] == "interface" else None,
        "before_bits_hex": f"0x{before:04x}", "after_bits_hex": f"0x{after:04x}",
        "xor_mask_hex": f"0x{mask:04x}",
        "before_value": before_value, "after_value": after_value,
        "before_value_class": before_class, "after_value_class": after_class,
        "original_sha256_before": before_hash, "original_sha256_after": after_hash,
        "original_tensor_unchanged": True, "replacement_uses_private_storage": True,
    }


def create_draft_fault_hook(config):
    """Install Draft-only point hooks after the first unarmed warmup forward."""
    arm_path = Path(config["arm_path"])
    installed = False
    active = {}
    counters = {}
    fired = set()

    def maybe_inject(point, tensor):
        context = base._context(config)
        if context is None or not active or active["batch"].forward_mode.is_idle():
            return tensor, False
        applied = False
        if arm_path.is_file():
            command = json.loads(arm_path.read_text())
            validate_command(command)
            if command["rid"] != context["rid"]:
                raise RuntimeError("Draft arm command belongs to a different active request")
            if command["point"] == point and active["mode"] in PHASES[command["phase"]]:
                fault_id = command["fault_id"]
                if fault_id in fired:
                    raise RuntimeError("fault_id was already injected; choose a new id")
                counters[fault_id] = counters.get(fault_id, 0) + 1
                if counters[fault_id] == int(command.get("occurrence", 1)):
                    event_path = Path(command["event_path"])
                    if event_path.exists():
                        raise FileExistsError(event_path)
                    tensor, evidence = copy_and_flip(tensor, command, active["hidden_size"])
                    event = {**evidence, "status": "injected", "fault_kind": "draft_activation_bitflip",
                             "fault_id": fault_id, "role": "draft", "point": point,
                             "module": MODULES[point],
                             "rid": context["rid"], "sample_id": context["sample_id"],
                             "phase": command["phase"], "forward_mode": active["mode"],
                             "eligible_occurrence": counters[fault_id],
                             "logical_token_position": int(active["positions"][evidence["row"]]),
                             "input_token_id": int(active["tokens"][evidence["row"]]),
                             "bit_positions": list(_parse_bit_positions(command["bit_positions"])),
                             "worker_pid": os.getpid(), "timestamp_unix": time.time(),
                             "target_parameters_modified": False,
                             "event_path": str(event_path)}
                    _atomic_write_json(event_path, event)
                    base._write(context, {"kind": "fault_context", "event": event})
                    arm_path.unlink()
                    fired.add(fault_id)
                    applied = True
        # Record the actual input/output consumed by Draft, also for clean runs.
        base._write(context, {
            "kind": "draft_path_state", "role": "draft", "point": point,
            "forward_mode": active["mode"], "fault_applied": applied,
            "hidden": base.summarize_hidden(tensor, active["positions"],
                                             last_only=base.prefill_last_only(active["batch"], tensor.shape[0])),
        })
        return tensor, applied

    def before_model(module, args):
        active.update(batch=args[2], mode=args[2].forward_mode.name,
                      positions=args[1], tokens=args[0], hidden_size=module.config.hidden_size)

    def interface_before(_module, args):
        tensor, applied = maybe_inject("interface", args[0])
        return (tensor, *args[1:]) if applied else None

    def output_hook(point):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor, applied = maybe_inject(point, tensor)
            if not applied:
                return None
            return (tensor, *output[1:]) if isinstance(output, tuple) else tensor
        return hook

    def installer(module, _args, _output):
        nonlocal installed
        if installed or int(module.config.num_hidden_layers) != 1 or not hasattr(module, "fc"):
            return None
        if base._context(config) is not None or arm_path.exists():
            raise RuntimeError("Draft fault hooks must install during unarmed warmup")
        module.register_forward_pre_hook(before_model)
        module.fc.register_forward_pre_hook(interface_before)
        module.fc.register_forward_hook(output_hook("fusion"))
        module.get_submodule("layers.0.mlp.down_proj").register_forward_hook(output_hook("decoder_down_proj"))
        installed = True
        return None

    return installer
