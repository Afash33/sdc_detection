"""Direct BF16 KV storage faults and FA3 consumption/lifetime evidence.

Installed through existing forward hooks. Qwen3/EAGLE3, TP=1, page_size=1,
eager, sequential requests only. No weight or installed-package modification.
"""
from __future__ import annotations

import functools
import importlib
import json
import os
from pathlib import Path
from types import MethodType

import torch

draft = importlib.import_module("fault_injection.2026-10-09_sglang_draft_bitflip")
base = draft.base
_SESSIONS = {}


def validate_command(command):
    for key in ("fault_id", "rid", "event_path"):
        if not command.get(key):
            raise ValueError(f"{key} required")
    if command.get("role") not in ("target", "draft") or command.get("region") not in ("history", "candidate"):
        raise ValueError("role target/draft and region history/candidate required")
    if command.get("component") not in ("key", "value"):
        raise ValueError("component key/value required")
    if command.get("bit_positions") is None or len(command["bit_positions"]) != 1:
        raise ValueError("one BF16 bit required")
    bit = command["bit_positions"][0]
    if not isinstance(bit, int) or not 0 <= bit < 16:
        raise ValueError("BF16 bit must be in [0,16)")
    for key in ("layer", "head", "dimension", "candidate_row"):
        if int(command.get(key, 0)) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if int(command.get("occurrence", 1)) < 1:
        raise ValueError("occurrence must be positive")


def select_slot(batch, loc, command):
    if batch.req_pool_indices.numel() != 1:
        raise ValueError("KV fault requires one request")
    if command["region"] == "candidate":
        row = int(command.get("candidate_row", 0))
        if row >= loc.numel():
            raise IndexError("candidate row outside current write")
        return int(loc[row]), row, int(batch.positions[row])
    length = int(batch.seq_lens[0])
    position = int(command.get("history_position", -1))
    if position < 0:
        position += length
    if not 0 <= position < length:
        raise IndexError("history position outside cached prefix")
    req_index = int(batch.req_pool_indices[0])
    slot = int(batch.req_to_token_pool.req_to_token[req_index, position])
    if bool((loc == slot).any()):
        raise ValueError("history slot overlaps current candidate writes")
    return slot, None, position


def flip_storage(pool, layer, slot, command):
    if getattr(pool, "layer_transfer_counter", None) is not None:
        raise ValueError("offloaded KV unsupported")
    k, v = pool._get_key_buffer(layer), pool._get_value_buffer(layer)
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16 or k.ndim != 3 or v.ndim != 3:
        raise TypeError("requires BF16 [slots,heads,dimensions] K/V storage")
    tensor = k if command["component"] == "key" else v
    head, dimension = int(command.get("head", 0)), int(command.get("dimension", 0))
    if not 0 < slot < tensor.shape[0] or not 0 <= head < tensor.shape[1] or not 0 <= dimension < tensor.shape[2]:
        raise IndexError("KV slot/head/dimension outside storage (slot zero is dummy)")
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    before_k, before_v = k[slot].detach().clone(), v[slot].detach().clone()
    bits = tensor.view(torch.int16)
    old = int(bits[slot, head, dimension]) & 0xffff
    before_value, before_class = draft._number(float(tensor[slot, head, dimension]))
    mask = 1 << command["bit_positions"][0]
    with torch.no_grad():
        bits[slot, head, dimension] = draft._signed_int16(old ^ mask)
    new = int(bits[slot, head, dimension]) & 0xffff
    after_value, after_class = draft._number(float(tensor[slot, head, dimension]))
    changes_k = int((before_k.view(torch.int16) != k[slot].view(torch.int16)).sum())
    changes_v = int((before_v.view(torch.int16) != v[slot].view(torch.int16)).sum())
    expected = (1, 0) if command["component"] == "key" else (0, 1)
    if new != old ^ mask or (changes_k, changes_v) != expected:
        raise RuntimeError("KV XOR/single-element isolation failed")
    return {"before_bits_hex": f"0x{old:04x}", "after_bits_hex": f"0x{new:04x}",
        "xor_mask_hex": f"0x{mask:04x}", "before_value": before_value, "after_value": after_value,
        "before_value_class": before_class, "after_value_class": after_class,
        "tensor_shape": list(tensor.shape), "tensor_dtype": str(tensor.dtype),
        "key_elements_changed": changes_k, "value_elements_changed": changes_v,
        "key_row_sha256_before": draft._digest(before_k), "key_row_sha256_after": draft._digest(k[slot]),
        "value_row_sha256_before": draft._digest(before_v), "value_row_sha256_after": draft._digest(v[slot]),
        "direct_storage_modified": True, "parameters_modified": False}


def visible_rows(metadata, slot):
    """Logical query rows whose actual FA3 table includes the live slot."""
    table = metadata.page_table.detach().cpu().tolist()
    lengths = metadata.cache_seqlens_int32.detach().cpu().tolist()
    offsets = metadata.cu_seqlens_q.detach().cpu().tolist()
    if len(table) != len(lengths) or len(offsets) != len(table) + 1:
        raise ValueError("FA3 metadata dimensions do not align")
    rows = []
    for index, (entries, length) in enumerate(zip(table, lengths)):
        if slot in entries[:length]:
            rows.extend(range(offsets[index], offsets[index + 1]))
    return sorted(set(rows))


class KVSession:
    def __init__(self, config):
        self.config = config
        self.pools, self.active, self.indices, self.installed = {}, {}, {}, set()
        self.watch = None
        self.rid = None
        self.counters, self.fired = {}, set()
        self.verify_installed = False

    def bits(self, pool, event):
        tensor = (pool._get_key_buffer(event["layer"]) if event["component"] == "key"
                  else pool._get_value_buffer(event["layer"]))
        return f"0x{int(tensor.view(torch.int16)[event['slot'],event['head'],event['dimension']]) & 0xffff:04x}"

    def log(self, kind, **fields):
        if self.watch is not None:
            base._write(self.watch["context"], {"kind": kind,
                "fault_id": self.watch["event"]["fault_id"], **fields})

    def install_pool(self, pool, role):
        self.pools[role] = pool
        if getattr(pool, "_sdc_direct_kv_fault", False):
            return
        original_write, original_read = pool.set_kv_buffer, pool.get_kv_buffer

        def write(_pool, layer, loc, cache_k, cache_v, *args, **kwargs):
            active = self.active.get(id(pool))
            override = kwargs.get("layer_id_override", args[2] if len(args) > 2 else None)
            layer_id = layer.layer_id if override is None else override
            watch = self.watch
            overwrites = (active and watch and watch["pool"] is pool and
                watch["event"]["layer"] == layer_id and bool((loc == watch["event"]["slot"]).any()))
            result = original_write(layer, loc, cache_k, cache_v, *args, **kwargs)
            if overwrites:
                self.log("kv_cache_lifecycle", action="rewritten", role=role, layer=layer_id,
                    slot=watch["event"]["slot"], forward_mode=active["batch"].forward_mode.name,
                    forward_index=active["index"], bits_after_write=self.bits(pool, watch["event"]),
                    original_fault_bits=watch["event"]["after_bits_hex"])
                if watch["event"]["region"] == "candidate":
                    watch["valid"] = False
            context = active["context"] if active else None
            arm = Path(self.config["arm_path"])
            if context is None or not arm.is_file():
                return result
            command = json.loads(arm.read_text())
            validate_command(command)
            if command["rid"] != context["rid"]:
                raise RuntimeError("KV arm belongs to another request")
            expected_mode = "TARGET_VERIFY" if command["role"] == "target" else "DECODE"
            if role != command["role"] or layer_id != command["layer"] or active["batch"].forward_mode.name != expected_mode:
                return result
            fid = command["fault_id"]
            if fid in self.fired:
                raise RuntimeError("KV fault already fired")
            self.counters[fid] = self.counters.get(fid, 0) + 1
            if self.counters[fid] != command.get("occurrence", 1):
                return result
            event_path = Path(command["event_path"])
            if event_path.exists():
                raise FileExistsError(event_path)
            slot, row, position = select_slot(active["batch"], loc, command)
            other_role = "draft" if role == "target" else "target"
            other = self.pools.get(other_role)
            if other is None or other is pool:
                raise RuntimeError("both independent KV pools must be installed before injection")
            other_layer = 0 if other_role == "draft" else layer_id
            # Other Target layer 0 for a Draft-layer-0 fault; no allocator.get_kvcache alias.
            other_k, other_v = other._get_key_buffer(other_layer), other._get_value_buffer(other_layer)
            other_hash = (draft._digest(other_k[slot]), draft._digest(other_v[slot]))
            evidence = flip_storage(pool, layer_id, slot, command)
            other_after = (draft._digest(other_k[slot]), draft._digest(other_v[slot]))
            if other_hash != other_after:
                raise RuntimeError("other role cache row changed directly")
            event = {**command, **evidence, "status": "injected", "fault_kind": "kv_storage_bitflip",
                "sample_id": context["sample_id"], "worker_pid": os.getpid(), "slot": slot,
                "candidate_row": row, "logical_position": position, "forward_index": active["index"],
                "forward_mode": active["batch"].forward_mode.name,
                "pool_identity": id(pool), "other_pool_identity": id(other),
                "other_role_row_unchanged": True, "other_role": other_role,
                "other_role_layer_checked": other_layer, "other_role_row_sha256_before": other_hash,
                "other_role_row_sha256_after": other_after,
                "eligible_occurrence": self.counters[fid]}
            draft._atomic_write_json(event_path, event)
            base._write(context, {"kind": "fault_context", "event": event})
            self.watch = {"event": event, "context": context, "pool": pool, "valid": True}
            self.fired.add(fid)
            arm.unlink()
            return result

        def read(_pool, layer_id):
            result = original_read(layer_id)
            active = self.active.get(id(pool))
            watch = self.watch
            if (active and active.get("in_attention") and watch and watch["valid"] and
                    watch["pool"] is pool and watch["event"]["layer"] == layer_id):
                batch = active["batch"]
                backend = batch.attn_backend
                prefix = visible_rows(backend.forward_metadata, watch["event"]["slot"])
                cascade = batch.forward_mode.name in {"DECODE", "TARGET_VERIFY"}
                expand = (visible_rows(backend.forward_metadata_spec_decode_expand, watch["event"]["slot"])
                          if cascade else [])
                self.log("kv_cache_read", role=role, layer=layer_id, slot=watch["event"]["slot"],
                    forward_index=active["index"], forward_mode=batch.forward_mode.name,
                    prefix_query_rows=prefix, candidate_query_rows=expand,
                    visible_query_rows=sorted(set(prefix + expand)), bits_at_read=self.bits(pool, watch["event"]),
                    source="actual get_kv_buffer inside FA3 attention; table membership, not a kernel load counter")
            return result

        pool.set_kv_buffer = MethodType(write, pool)
        pool.get_kv_buffer = MethodType(read, pool)
        pool._sdc_direct_kv_fault = True

    def install_backend(self, backend, pool, role):
        if getattr(backend, "_sdc_kv_attention_trace", False):
            return
        if (type(backend).__name__ != "FlashAttentionBackend" or backend.page_size != 1
                or backend.topk != 4 or backend.fa_impl_ver != 3):
            raise TypeError("KV experiment requires FA3, page_size=1, tree topk=4")
        for name in ("forward_extend", "forward_decode"):
            original = getattr(backend, name)

            def wrap(_backend, q, k, v, layer, batch, *args, _original=original, **kwargs):
                active = self.active.get(id(pool))
                if active:
                    active["in_attention"] = True
                try:
                    output = _original(q, k, v, layer, batch, *args, **kwargs)
                finally:
                    if active:
                        active["in_attention"] = False
                if active and active["context"] and layer.layer_id == (20 if role == "target" else 0):
                    base._write(active["context"], {"kind": "kv_attention_state", "role": role,
                        "layer": layer.layer_id, "forward_index": active["index"],
                        "forward_mode": batch.forward_mode.name,
                        "hidden": base.summarize_hidden(output.reshape(output.shape[0], -1),
                            batch.positions, last_only=base.prefill_last_only(batch, output.shape[0]))})
                return output

            setattr(backend, name, MethodType(wrap, backend))
        backend._sdc_kv_attention_trace = True

    def before(self, module, args):
        batch = args[2]
        context = base._context(self.config)
        if context and context["rid"] != self.rid:
            self.rid, self.indices, self.watch = context["rid"], {}, None
        role = "draft" if int(module.config.num_hidden_layers) == 1 else "target"
        pool = batch.token_to_kv_pool
        self.install_pool(pool, role)
        self.install_backend(batch.attn_backend, pool, role)
        index = self.indices.get(role, 0)
        self.indices[role] = index + 1
        self.active[id(pool)] = {"context": context, "batch": batch, "index": index}

    def after(self, module, args, output):
        self.active.pop(id(args[2].token_to_kv_pool), None)

    def verify_lifecycle(self, batch, original_slots, result):
        watch = self.watch
        if watch is None or watch["context"]["rid"] != self.rid:
            return
        event = watch["event"]
        indices = result.accept_indices.detach().cpu().tolist()
        retained = [original_slots[i] for i in indices]
        if event["role"] == "target" and event["region"] == "candidate" and event["slot"] in original_slots:
            accepted = event["slot"] in retained
            self.log("kv_cache_lifecycle", action="retained_as_history" if accepted else "discarded",
                role="target", slot=event["slot"], accepted_row_indices=indices,
                accepted_slots=retained, bits_after_verify=self.bits(watch["pool"], event))
            watch["valid"] = accepted
        elif event["role"] == "draft" and event["region"] == "candidate":
            self.log("kv_cache_lifecycle", action="await_draft_recompute", role="draft", slot=event["slot"],
                same_slot_selected_for_target_history=event["slot"] in retained,
                accepted_row_indices=indices, accepted_slots=retained)

    def install_verify(self):
        if self.verify_installed:
            return
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        original = EagleVerifyInput.verify

        @functools.wraps(original)
        def verify(info, batch, logits, *args, **kwargs):
            context = base._context(self.config)
            if context is None or batch.forward_mode.is_idle():
                return original(info, batch, logits, *args, **kwargs)
            if len(batch.reqs) != 1 or batch.reqs[0].rid != context["rid"]:
                raise RuntimeError("KV lifecycle requires exactly the armed request")
            slots = batch.out_cache_loc.detach().cpu().tolist()
            result = original(info, batch, logits, *args, **kwargs)
            self.verify_lifecycle(batch, slots, result)
            return result

        EagleVerifyInput.verify = verify
        self.verify_installed = True


def create_kv_fault_hook(config):
    key = (config["context_path"], config["arm_path"])
    session = _SESSIONS.setdefault(key, KVSession(config))

    def installer(module, args, output):
        if id(module) in session.installed:
            return None
        if int(module.config.num_hidden_layers) not in (1, 36):
            raise TypeError("only Qwen3-8B Target and one-layer EAGLE3 Draft supported")
        if base._context(config) is not None or Path(config["arm_path"]).exists():
            raise RuntimeError("KV hooks must install during unarmed warmup")
        role = "draft" if int(module.config.num_hidden_layers) == 1 else "target"
        session.install_pool(args[2].token_to_kv_pool, role)
        if role == "target":
            session.install_verify()
        module.register_forward_pre_hook(session.before)
        module.register_forward_hook(session.after)
        session.installed.add(id(module))
        return None

    return installer


def check_events(record, events):
    errors = []
    event = record.get("kv_fault_event")
    injected = [e for e in events if e["kind"] == "fault_context" and e.get("event", {}).get("fault_kind") == "kv_storage_bitflip"]
    if event is None:
        return ["unexpected KV injection"] if injected else []
    before, after, mask = (int(event[k], 16) for k in ("before_bits_hex", "after_bits_hex", "xor_mask_hex"))
    expected = (1, 0) if event["component"] == "key" else (0, 1)
    if (before ^ mask != after or (event["key_elements_changed"],event["value_elements_changed"]) != expected
            or not event["direct_storage_modified"] or not event["other_role_row_unchanged"]
            or event["other_role_row_sha256_before"] != event["other_role_row_sha256_after"]
            or event["parameters_modified"] or event["pool_identity"] == event["other_pool_identity"]
            or event["rid"] != record["request"]["rid"] or event["sample_id"] != record["sample_id"]
            or len(injected) != 1):
        errors.append("invalid KV XOR/isolation/request/single-fire evidence")
    reads = [e for e in events if e["kind"] == "kv_cache_read" and e["fault_id"] == event["fault_id"]]
    if not any(e["visible_query_rows"] and e["bits_at_read"] == event["after_bits_hex"] for e in reads):
        errors.append("no FA3-visible read of injected KV bits")
    if event["region"] == "candidate":
        lifetime = [e for e in events if e["kind"] == "kv_cache_lifecycle" and e["fault_id"] == event["fault_id"]]
        required = {"retained_as_history", "discarded"} if event["role"] == "target" else {"await_draft_recompute"}
        if not any(e["action"] in required for e in lifetime):
            errors.append("missing candidate KV lifetime decision")
    if any(e["worker_pid"] != event["worker_pid"] for e in events):
        errors.append("KV event worker PID differs from trace")
    return errors
