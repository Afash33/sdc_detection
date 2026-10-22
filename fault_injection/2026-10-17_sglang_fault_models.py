"""Request-scoped Target result/state faults and bounded EAGLE3 telemetry.

Loaded by SGLang --forward-hooks; does not edit SGLang or model files.
Weight control delegates to the existing, bit-verified Target custom loader.
Output and KV events use the same explicitly seeded BF16 damage primitive.
No complete hidden, logits, or KV tensor is written to disk.
"""
from __future__ import annotations

import functools
import importlib
import json
import random
from pathlib import Path
from types import MethodType

import torch

trace = importlib.import_module("experiments.2026-10-06_sglang_baseline_trace")
weights = importlib.import_module("fault_injection.sglang_weight_bitflip")
oldkv = importlib.import_module("fault_injection.2026-10-10_sglang_kv_bitflip")
PROJECTIONS = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")


def projection_component(name, module, column, width):
    """Label a fused output using the actual TP=1 module partition sizes."""
    labels = ("q", "k", "v") if name.endswith("qkv_proj") else (
        ("gate", "up") if name.endswith("gate_up_proj") else ())
    sizes = getattr(module, "output_partition_sizes", None)
    if not sizes or len(sizes) != len(labels) or sum(sizes) != width:
        return "unsplit" if not labels else "unknown_partition"
    offset = 0
    for label, size in zip(labels, sizes):
        if column < offset + size:
            return label
        offset += size
    raise IndexError("column outside fused projection output")


def damage(tensor, command):
    """Change only explicit selected cells; return exact before/after bit evidence."""
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2 or not tensor.numel():
        raise TypeError("damage requires a nonempty 2-D BF16 tensor")
    rng = random.Random(command["seed"])
    pattern = command["pattern"]
    side = int(command.get("side", 1)) if pattern.startswith("region_") else 1
    if min(tensor.shape) < side:
        raise ValueError("requested region does not fit tensor")
    row = int(command.get("row", rng.randrange(tensor.shape[0] - side + 1)))
    col = int(command.get("column", rng.randrange(tensor.shape[1] - side + 1)))
    if not (0 <= row <= tensor.shape[0]-side and 0 <= col <= tensor.shape[1]-side):
        raise IndexError("selected damage coordinates outside tensor")
    if pattern == "bit":
        positions = list(command["bit_positions"])
        if len(positions) not in (1, 2) or len(set(positions)) != len(positions) or any(
                not isinstance(b, int) or not 0 <= b < 16 for b in positions):
            raise ValueError("one/two distinct BF16 bits required")
    elif pattern not in {"word_random", "region_random", "region_zero"}:
        raise ValueError("unsupported damage pattern")
    bits = tensor.view(torch.int16)
    evidence = []
    for r in range(row, row + side):
        for c in range(col, col + side):
            before = int(bits[r, c]) & 0xffff
            if pattern == "bit":
                after = before ^ sum(1 << b for b in positions)
            elif pattern == "region_zero":
                after = 0
            else:
                after = rng.randrange(65536)
                while after == before:
                    after = rng.randrange(65536)
            with torch.no_grad():
                bits[r, c] = weights._signed_int16(after)
            actual = int(bits[r, c]) & 0xffff
            if actual != after:
                raise RuntimeError("damage bit readback failed")
            evidence.append({"row": r, "column": c, "before": f"{before:04x}",
                             "after": f"{actual:04x}", "xor": f"{before ^ actual:04x}"})
    return {"cells": evidence, "changed_elements": sum(x["before"] != x["after"] for x in evidence),
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "pattern": pattern, "side": side}


def control_loader(model, named_tensors):
    """Inventory or delegate existing verified inject/restore control, Target only."""
    command = weights._one_command(named_tensors)
    if command["action"] == "inventory":
        inventory = [{"module": name, "shape": list(m.weight.shape), "numel": m.weight.numel(),
                      "dtype": str(m.weight.dtype)} for name, m in model.named_modules()
                     if any(name.endswith(p) for p in PROJECTIONS)]
        weights._atomic_write_json(Path(command["event_path"]), {"modules": inventory})
    elif command["action"] == "inject":
        weights._inject(model, command)
    elif command["action"] == "restore":
        weights._restore(model, command)
    else:
        raise ValueError("unsupported control command")


class Session:
    def __init__(self, config):
        self.config = config
        self.installed = set()
        self.rid = None
        self.command = None
        self.context = None
        self.active = None
        self.occurrence = 0
        self.fired = False
        self.watch = None
        self.verify_installed = False
        self.forward_counts = {}

    def emit(self, kind, **fields):
        if self.context:
            trace._write(self.context, {"kind": kind, **fields})

    def before(self, module, args):
        role = "draft" if module.config.num_hidden_layers == 1 else "target"
        context = trace._context(self.config)
        if context and self.rid != context["rid"]:
            if self.command and not self.fired:
                raise RuntimeError("previous request ended before scheduled injection")
            self.context, self.rid = context, context["rid"]
            arm = Path(self.config["arm_path"])
            self.command = json.loads(arm.read_text()) if role == "target" and arm.exists() else None
            if self.command and self.command["rid"] != self.rid:
                raise RuntimeError("arm/request ID mismatch")
            self.fired, self.occurrence, self.watch = False, 0, None
            self.forward_counts = {}
        self.context = context
        batch = args[2]
        self.forward_counts[role] = self.forward_counts.get(role, 0) + 1
        if role == "target":
            self.active = {"batch": batch, "positions": args[1],
                           "index": self.forward_counts[role] - 1}
            self.install_pool(batch.token_to_kv_pool)
            self.install_backend(batch.attn_backend)

    def eligible(self, kind, module=None, layer=None):
        c = self.command
        if not self.context or not c or self.fired or not self.active or c["kind"] != kind:
            return False
        modes = {"prefill": {"EXTEND", "MIXED", "SPLIT_PREFILL"},
                 "target_verify": {"TARGET_VERIFY"}}
        if self.active["batch"].forward_mode.name not in modes[c["phase"]]:
            return False
        if module is not None and c["module"] != module:
            return False
        if layer is not None and c["layer"] != layer:
            return False
        self.occurrence += 1
        return self.occurrence == c["occurrence"]

    def record_fault(self, evidence, **fields):
        c = self.command
        event = {**c, **fields, "evidence": evidence, "status": "injected",
                 "forward_mode": self.active["batch"].forward_mode.name,
                 "forward_index": self.active["index"], "eligible_occurrence": self.occurrence}
        path = Path(c["event_path"])
        if path.exists():
            raise FileExistsError(path)
        weights._atomic_write_json(path, event)
        self.emit("fault_context", event=event)
        self.fired = True
        Path(self.config["arm_path"]).unlink()
        return event

    def result_hook(self, name):
        def hook(module, args, output):
            if not self.eligible("compute", module=name):
                return None
            tensor = output[0] if isinstance(output, tuple) else output
            # This copy is the *consumed result*, leaving any aliased producer buffer alone.
            changed = tensor.clone()
            evidence = damage(changed, self.command)
            cell = evidence["cells"][0]
            positions = self.active.get("positions")
            coordinate = {"row": cell["row"], "column": cell["column"],
                          "token_position": int(positions[cell["row"]]) if positions is not None else None,
                          "projection_component": projection_component(name, module, cell["column"], tensor.shape[-1])}
            self.record_fault(evidence, module=name, source="operator_output",
                              parameters_modified=False, output_coordinate=coordinate)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        return hook

    def bits(self):
        w = self.watch
        t = (w["pool"]._get_key_buffer(w["layer"]) if w["component"] == "key"
             else w["pool"]._get_value_buffer(w["layer"]))
        return f"{int(t.view(torch.int16)[w['slot'], w['head'], w['dimension']]) & 0xffff:04x}"

    def install_pool(self, pool):
        if getattr(pool, "_sdc_models_pool", False):
            return
        write, read = pool.set_kv_buffer, pool.get_kv_buffer

        def write_wrapper(_pool, layer, loc, k, v, *args, **kwargs):
            lid = kwargs.get("layer_id_override", args[2] if len(args) > 2 else None)
            lid = layer.layer_id if lid is None else lid
            result = write(layer, loc, k, v, *args, **kwargs)
            w = self.watch
            if self.context and w and w["valid"] and w["pool"] is pool and w["layer"] == lid:
                if bool((loc == w["slot"]).any()):
                    self.emit("kv_lifecycle", action="overwritten", slot=w["slot"], layer=lid)
                    w["valid"] = False
            if not self.eligible("kv", layer=lid):
                return result
            c = self.command
            rng = random.Random(c["seed"])
            selector = {"region": c["region"], "candidate_row": rng.randrange(loc.numel()),
                        "history_position": rng.randrange(int(self.active["batch"].seq_lens[0]))}
            slot, row, position = oldkv.select_slot(self.active["batch"], loc, selector)
            tensor = pool._get_key_buffer(lid) if c["component"] == "key" else pool._get_value_buffer(lid)
            if not 0 < slot < tensor.shape[0]:
                raise IndexError("KV slot is dummy or outside real pool")
            h, d = rng.randrange(tensor.shape[1]), rng.randrange(tensor.shape[2])
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
            # Direct view: mutate actual stored row, never a detached copy.
            evidence = damage(tensor[slot].reshape(1, -1), {**c, "row": 0,
                                                        "column": h*tensor.shape[2]+d})
            self.watch = {"pool": pool, "slot": slot, "layer": lid, "head": h,
                          "dimension": d, "component": c["component"], "valid": True,
                          "region": c["region"]}
            self.record_fault(evidence, slot=slot, head=h, dimension=d,
                              logical_position=position, candidate_row=row,
                              direct_storage_modified=True, parameters_modified=False)
            return result

        def read_wrapper(_pool, lid):
            result = read(lid)
            w = self.watch
            if self.context and w and w["valid"] and w["layer"] == lid and self.active:
                batch = self.active["batch"]
                if getattr(self, "in_attention", False):
                    backend = batch.attn_backend
                    rows = oldkv.visible_rows(backend.forward_metadata, w["slot"])
                    if batch.forward_mode.name == "TARGET_VERIFY":
                        rows += oldkv.visible_rows(backend.forward_metadata_spec_decode_expand, w["slot"])
                    self.emit("kv_consumed", layer=lid, slot=w["slot"],
                              forward_index=self.active["index"], bits=self.bits(),
                              logical_visible_query_rows=sorted(set(rows)),
                              note="actual FA3 get_kv_buffer; logical visibility, not physical load count")
            return result
        pool.set_kv_buffer = MethodType(write_wrapper, pool)
        pool.get_kv_buffer = MethodType(read_wrapper, pool)
        pool._sdc_models_pool = True

    def install_backend(self, backend):
        if getattr(backend, "_sdc_models_attention", False):
            return
        if type(backend).__name__ != "FlashAttentionBackend" or backend.fa_impl_ver != 3 or backend.page_size != 1:
            raise TypeError("this bounded KV evidence requires FA3 with page_size=1")
        for name in ("forward_extend", "forward_decode"):
            original = getattr(backend, name)
            def wrapped(_backend, *args, _original=original, **kwargs):
                self.in_attention = True
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.in_attention = False
            setattr(backend, name, MethodType(wrapped, backend))
        backend._sdc_models_attention = True

    def install_verify(self):
        if self.verify_installed:
            return
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        original = EagleVerifyInput.verify
        @functools.wraps(original)
        def verify(info, batch, logits, *args, **kwargs):
            context = trace._context(self.config)
            if not context or batch.forward_mode.is_idle():
                return original(info, batch, logits, *args, **kwargs)
            if len(batch.reqs) != 1 or batch.reqs[0].rid != context["rid"]:
                raise RuntimeError("verifier/request scope mismatch")
            req = batch.reqs[0]
            start = len(req.output_ids)
            slots = batch.out_cache_loc.detach().cpu().tolist()
            event = {"tree_tokens": trace._tolist(info.draft_token),
                     "tree_positions": trace._tolist(info.positions),
                     "tree_next_token": trace._tolist(info.retrieve_next_token),
                     "tree_next_sibling": trace._tolist(info.retrieve_next_sibling),
                     "probabilities": trace.summarize_logits(logits.next_token_logits),
                     "generated_index_before": start}
            result = original(info, batch, logits, *args, **kwargs)
            indices = trace._tolist(result.accept_indices)
            self.emit("verify_block", **event, accepted_rows=indices,
                      committed_token_ids=list(req.output_ids[start:]),
                      generated_index_after=len(req.output_ids))
            w = self.watch
            if w and w["valid"] and w["region"] == "candidate" and w["slot"] in slots:
                accepted = w["slot"] in [slots[i] for i in indices]
                self.emit("kv_lifecycle", action="retained_as_history" if accepted else "discarded",
                          slot=w["slot"], layer=w["layer"], bits=self.bits())
                w["valid"] = accepted
                if accepted:
                    w["region"] = "history"
            return result
        EagleVerifyInput.verify = verify
        self.verify_installed = True

    def after(self, module, args, output):
        role = "draft" if module.config.num_hidden_layers == 1 else "target"
        if not self.context:
            return None
        batch = args[2]
        if batch.forward_mode.is_idle():
            return None
        hidden, aux = output if isinstance(output, tuple) else (output, [])
        last_only = trace.prefill_last_only(batch, hidden.shape[0])
        self.emit("model_hidden", role=role, forward_index=self.forward_counts[role]-1,
                  forward_mode=batch.forward_mode.name,
                  hidden=trace.summarize_hidden(hidden, args[1], last_only=last_only, slice_size=16),
                  taps=[trace.summarize_hidden(t, args[1], last_only=last_only, slice_size=16) for t in aux])
        return None


def create_fault_model_hook(config):
    session = Session(config)
    def install(module, args, output):
        if id(module) in session.installed:
            return None
        if trace._context(config) or Path(config["arm_path"]).exists():
            raise RuntimeError("install only during unarmed warmup")
        role = "draft" if module.config.num_hidden_layers == 1 else "target"
        if role == "target":
            for name, child in module.named_modules():
                if any(name.endswith(p) for p in PROJECTIONS):
                    child.register_forward_hook(session.result_hook("model." + name))
            session.install_verify()
        module.register_forward_pre_hook(session.before)
        module.register_forward_hook(session.after)
        session.installed.add(id(module))
        return None
    return install


def create_draft_probability_hook(config):
    """Draft reduced-vocabulary top-8/entropy; Target probabilities are in verify trace."""
    def hook(module, inputs, output):
        context = trace._context(config)
        if context is None or output.next_token_logits is None:
            return None
        logits = output.next_token_logits
        if logits.shape[-1] == 32000:
            trace._write(context, {"kind": "draft_probabilities", "role": "draft",
                         "token_id_space": "draft_32000_vocab_not_target_ids",
                         "forward_mode": inputs[3].forward_mode.name,
                         "probabilities": trace.summarize_logits(logits)})
        return None
    return hook
