"""Request-scoped Target output faults; loaded by SGLang, not a launch script.

final_norm: BF16 normalized hidden output; lm_head: raw BF16 matmul output;
logits: FP32 processed logits, before sampling and logprob computation.
Wrap only this worker's processor methods; installed package files stay intact.
Supported experiment: Qwen3-8B / EAGLE3, TP=1, eager, sequential requests.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import torch

from fault_injection.sglang_activation_bitflip import _atomic_write_json

draft = importlib.import_module("fault_injection.2026-10-09_sglang_draft_bitflip")
base = draft.base
POINTS = ("final_norm", "lm_head", "logits")
PHASES = {"prefill": {"EXTEND", "MIXED", "SPLIT_PREFILL"},
          "target_verify": {"TARGET_VERIFY"}}


def validate_command(command):
    if command.get("point") not in POINTS or command.get("phase") not in PHASES:
        raise ValueError("unsupported Target output point or phase")
    for key in ("fault_id", "rid", "event_path"):
        if not command.get(key):
            raise ValueError(f"{key} is required")
    bit = command.get("bit_positions")
    width = 32 if command["point"] == "logits" else 16
    if not isinstance(bit, list) or len(bit) != 1 or not isinstance(bit[0], int) or not 0 <= bit[0] < width:
        raise ValueError(f"exactly one bit in [0,{width}) is required")
    if int(command.get("channel", 0)) < 0 or int(command.get("occurrence", 1)) < 1:
        raise ValueError("channel must be nonnegative; occurrence positive")


def copy_and_flip(tensor, command):
    expected = torch.float32 if command["point"] == "logits" else torch.bfloat16
    if tensor.dtype != expected or tensor.ndim != 2 or not tensor.numel():
        raise TypeError(f"{command['point']} requires nonempty [rows,channels] {expected}")
    if expected == torch.bfloat16:
        return draft.copy_and_flip(tensor, command, tensor.shape[1])
    original_hash = draft._digest(tensor)
    changed = tensor.detach().clone(memory_format=torch.contiguous_format)
    row = int(command.get("row", -1))
    row = row + tensor.shape[0] if row < 0 else row
    channel = int(command.get("channel", 0))
    if not 0 <= row < tensor.shape[0] or not 0 <= channel < tensor.shape[1]:
        raise IndexError("Target logits row/channel out of bounds")
    bits = changed.view(torch.int32)
    before = int(bits[row, channel]) & 0xffffffff
    before_value, before_class = draft._number(float(changed[row, channel]))
    mask = 1 << command["bit_positions"][0]
    value = before ^ mask
    with torch.no_grad():
        bits[row, channel] = value if value < 1 << 31 else value - (1 << 32)
    after = int(bits[row, channel]) & 0xffffffff
    after_value, after_class = draft._number(float(changed[row, channel]))
    after_hash = draft._digest(tensor)
    if before ^ mask != after or original_hash != after_hash:
        raise RuntimeError("Target logits XOR/isolation check failed")
    return changed, {"tensor_shape": list(tensor.shape), "tensor_dtype": str(tensor.dtype),
        "row": row, "channel": channel, "flat_index": row * tensor.shape[1] + channel,
        "before_bits_hex": f"0x{before:08x}", "after_bits_hex": f"0x{after:08x}",
        "xor_mask_hex": f"0x{mask:08x}", "before_value": before_value,
        "after_value": after_value, "before_value_class": before_class,
        "after_value_class": after_class, "original_sha256_before": original_hash,
        "original_sha256_after": after_hash, "original_tensor_unchanged": True,
        "replacement_uses_private_storage": True}


def create_target_output_fault_hook(config):
    arm = Path(config["arm_path"])
    installed = set()
    active = {}
    counters, fired = {}, set()

    def transform(point, tensor):
        context = base._context(config)
        if context is None or not active or active["batch"].forward_mode.is_idle():
            return tensor
        command = json.loads(arm.read_text()) if arm.is_file() else None
        applied = False
        before = None
        if command:
            validate_command(command)
            if command["rid"] != context["rid"]:
                raise RuntimeError("Target output arm belongs to a different request")
            if point == command["point"] and active["batch"].forward_mode.name in PHASES[command["phase"]]:
                fid = command["fault_id"]
                if fid in fired:
                    raise RuntimeError("Target fault_id already injected")
                counters[fid] = counters.get(fid, 0) + 1
                if counters[fid] == int(command.get("occurrence", 1)):
                    event_path = Path(command["event_path"])
                    if event_path.exists():
                        raise FileExistsError(event_path)
                    before = base.summarize_hidden(tensor, last_only=point == "final_norm" and
                        base.prefill_last_only(active["batch"], tensor.shape[0])) if point == "final_norm" else base.summarize_logits(tensor)
                    tensor, evidence = copy_and_flip(tensor, command)
                    event = {**evidence, "status": "injected", "role": "target",
                        "fault_kind": "target_output_activation_bitflip", "point": point,
                        "fault_id": fid, "rid": context["rid"], "sample_id": context["sample_id"],
                        "phase": command["phase"], "forward_mode": active["batch"].forward_mode.name,
                        "eligible_occurrence": counters[fid], "bit_positions": command["bit_positions"],
                        "worker_pid": os.getpid(), "timestamp_unix": time.time(),
                        "parameters_modified": False, "target_auxiliary_features_modified_directly": False,
                        "row_domain": "model_hidden_rows" if point == "final_norm" else "pruned_logit_rows",
                        "channel_domain": "hidden_channel" if point == "final_norm" else "target_token_id",
                        "event_path": str(event_path)}
                    _atomic_write_json(event_path, event)
                    base._write(context, {"kind": "fault_context", "event": event})
                    arm.unlink()
                    fired.add(fid)
                    applied = True
        summary = (base.summarize_hidden(tensor, last_only=base.prefill_last_only(active["batch"], tensor.shape[0]))
                   if point == "final_norm" else base.summarize_logits(tensor))
        base._write(context, {"kind": "target_output_state", "role": "target", "point": point,
            "forward_mode": active["batch"].forward_mode.name, "fault_applied": applied,
            "state": summary, "before_fault_state": before})
        return tensor

    def before_model(_module, args):
        active["batch"] = args[2]

    def before_logits(_module, args):
        active["batch"] = args[3]

    def norm_hook(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        changed = transform("final_norm", tensor)
        if changed is tensor:
            return None
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def installer(module, _args, _output):
        if id(module) in installed:
            return None
        if hasattr(module, "layers"):
            if int(module.config.num_hidden_layers) != 36 or hasattr(module, "fc"):
                return None
            if base._context(config) is not None or arm.exists():
                raise RuntimeError("Target output hooks must install during unarmed warmup")
            module.register_forward_pre_hook(before_model)
            module.norm.register_forward_hook(norm_hook)
        elif hasattr(module, "_compute_lm_head"):
            if module.vocab_size != 151936:
                return None  # EAGLE3 Draft reduced vocabulary; never patch it.
            if base._context(config) is not None or arm.exists():
                raise RuntimeError("Target output hooks must install during unarmed warmup")
            module.register_forward_pre_hook(before_logits)
            compute, get_logits = module._compute_lm_head, module._get_logits
            def compute_wrapper(*args, **kwargs):
                return transform("lm_head", compute(*args, **kwargs))
            def logits_wrapper(*args, **kwargs):
                return transform("logits", get_logits(*args, **kwargs))
            module._compute_lm_head = compute_wrapper
            module._get_logits = logits_wrapper
        else:
            return None
        installed.add(id(module))
        return None

    return installer
