"""Read-only EAGLE3 telemetry for SGLang 0.5.12 (TP=1, CUDA graphs off).

Loaded through --forward-hooks, not run directly. No tensor is changed.
The client arms collection with a JSON context file, so warmups are excluded.
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

_VERIFY_INSTALLED = False


def _tolist(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _context(config: dict) -> dict | None:
    path = Path(config["context_path"])
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _write(context: dict, event: dict) -> None:
    event.update(sample_id=context["sample_id"], rid=context["rid"],
                 worker_pid=os.getpid(), timestamp_ns=time.time_ns())
    # TP=1 and sequential requests: a single worker appends these events.
    with open(context["trace_path"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False,
                                separators=(",", ":")) + "\n")


def prefill_last_only(batch, row_count: int) -> bool:
    # is_extend() also includes TARGET_VERIFY and DRAFT_EXTEND in SGLang.
    # Never discard candidate-tree rows. Draft prompt extension is long, whereas
    # draft extension after verification contains at most four committed rows.
    return batch.forward_mode.name in {"EXTEND", "MIXED", "SPLIT_PREFILL"} or (
        batch.forward_mode.name in {"DRAFT_EXTEND", "DRAFT_EXTEND_V2"}
        and row_count > 64
    )


def summarize_hidden(tensor: torch.Tensor, positions=None, *, last_only=False,
                     slice_size=128) -> dict:
    """Full-dimensional norms, plus a fixed evenly spaced diagnostic slice."""
    if tensor.ndim != 2:
        raise ValueError(f"expected [rows, hidden_size], got {tuple(tensor.shape)}")
    rows = tensor.shape[0]
    indices = [rows - 1] if last_only and rows else list(range(rows))
    x = tensor[indices].detach().float()
    finite = torch.isfinite(x)
    safe = torch.where(finite, x, 0.0)
    width = x.shape[-1]
    dimensions = torch.linspace(0, width - 1, min(slice_size, width),
                                device=x.device).long()
    sliced = x[:, dimensions].cpu().tolist()
    sliced = [[v if abs(v) != float("inf") and v == v else None for v in row]
              for row in sliced]
    # FP64 prevents the norm calculation itself overflowing for large BF16 faults.
    norm = torch.linalg.vector_norm(safe.double(), dim=-1).cpu().tolist()
    return {
        "original_shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "row_indices": indices,
        "positions": _tolist(positions[indices]) if positions is not None else None,
        "l2_full": norm,
        "rms_full": [v / width ** 0.5 for v in norm],
        "max_abs_finite": safe.abs().amax(dim=-1).cpu().tolist(),
        "nonfinite_count": (~finite).sum(dim=-1).cpu().tolist(),
        "slice_dimensions": dimensions.cpu().tolist(), "slice_values": sliced,
        "norm_excludes_nonfinite": True,
    }


def summarize_logits(logits: torch.Tensor, top_k=8) -> dict:
    # Exponent-bit faults can produce huge finite values: avoid FP32 overflow
    # when subtracting logsumexp or taking probability-weighted logs.
    x = logits.detach().double()
    finite_rows = torch.isfinite(x).all(dim=-1)
    safe = torch.where(torch.isfinite(x), x, 0.0)
    lp = torch.log_softmax(safe, dim=-1)
    values, ids = torch.topk(lp, min(top_k, x.shape[-1]), dim=-1)
    probabilities = lp.exp()
    entropy = -torch.where(probabilities > 0, probabilities * lp, 0.0).sum(dim=-1)
    valid = finite_rows.cpu().tolist()
    vals = values.cpu().tolist()
    return {
        "shape": list(x.shape), "finite_rows": valid,
        "nonfinite_count": (~torch.isfinite(x)).sum(dim=-1).cpu().tolist(),
        "top_token_ids": ids.cpu().tolist(),
        "top_logprobs": [row if ok else None for row, ok in zip(vals, valid)],
        "entropy_nats": [v if ok else None for v, ok in
                         zip(entropy.cpu().tolist(), valid)],
        "top1_top2_logit_margin": [row[0] - row[1] if ok and len(row) > 1 else None
                                  for row, ok in zip(vals, valid)],
        "distribution": "raw_logits_softmax_before_sampling_constraints",
        "invalid_rows_have_placeholder_top_ids": True,
    }


def _install_verify_trace(config: dict) -> None:
    """Wrap verification in memory; leave installed SGLang files untouched."""
    global _VERIFY_INSTALLED
    if _VERIFY_INSTALLED:
        return
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    original = EagleVerifyInput.verify

    @functools.wraps(original)
    def verify(self, batch, logits_output, *args, **kwargs):
        context = _context(config)
        if context is None or batch.forward_mode.is_idle():
            return original(self, batch, logits_output, *args, **kwargs)
        if len(batch.reqs) != 1 or batch.reqs[0].rid != context["rid"]:
            raise RuntimeError("trace requires exactly the armed request (TP=1)")
        req = batch.reqs[0]
        start_index = len(req.output_ids)
        event = {
            "kind": "verify_block", "block_index": req.spec_verify_ct,
            "generated_index_before": start_index,
            "draft_token_ids": _tolist(self.draft_token),
            "tree_positions": _tolist(self.positions),
            "retrieve_index": _tolist(self.retrieve_index),
            "retrieve_next_token": _tolist(self.retrieve_next_token),
            "retrieve_next_sibling": _tolist(self.retrieve_next_sibling),
            "tree_token_count_including_root": self.draft_token_num,
            "target_probabilities": summarize_logits(logits_output.next_token_logits),
        }
        result = original(self, batch, logits_output, *args, **kwargs)
        event.update(
            accepted_row_indices=_tolist(result.accept_indices),
            committed_token_ids=list(req.output_ids[start_index:]),
            generated_index_after=len(req.output_ids),
            num_correct_drafts_per_req=_tolist(result.num_correct_drafts_per_req_cpu),
            cumulative_correct_drafts=req.spec_num_correct_drafts,
            cumulative_verify_count=req.spec_verify_ct,
            note="committed tokens include target token; EOS/length can trim last block",
        )
        _write(context, event)
        return result

    EagleVerifyInput.verify = verify
    _VERIFY_INSTALLED = True


def create_model_trace_hook(config: dict):
    """Hook nested `model`, which is called through nn.Module.__call__.

    SGLang calls top-level model.forward directly; a root forward hook would
    miss it. Both Target and Draft have a nested module named `model`.
    """
    calls = 0
    intermediate_installed = False
    active_forward = {}

    def hook(module, inputs, output):
        nonlocal calls, intermediate_installed
        role = "draft" if int(module.config.num_hidden_layers) == 1 else "target"
        if role == "target":
            _install_verify_trace(config)
        if not intermediate_installed:
            # Install during unarmed warmup; future calls supply phase/positions.
            def pre_hook(_module, args):
                active_forward["batch"] = args[2]
                active_forward["positions"] = args[1]

            module.register_forward_pre_hook(pre_hook)
            if role == "target":
                point = module.get_submodule("layers.20.mlp.down_proj")

                def down_hook(_module, _args, value):
                    context = _context(config)
                    batch = active_forward.get("batch")
                    if context is None or batch is None:
                        return None
                    tensor = value[0] if isinstance(value, tuple) else value
                    _write(context, {
                        "kind": "intermediate", "role": "target",
                        "module": "model.layers.20.mlp.down_proj",
                        "forward_mode": batch.forward_mode.name,
                        "forward_index": calls,
                        "hidden": summarize_hidden(
                            tensor, active_forward["positions"],
                            last_only=prefill_last_only(batch, tensor.shape[0])),
                    })

                point.register_forward_hook(down_hook)
            intermediate_installed = True
        context = _context(config)
        if context is None:
            return None
        batch = inputs[2]
        if batch.forward_mode.is_idle():
            return None
        positions = inputs[1]
        hidden, auxiliary = output if isinstance(output, tuple) else (output, [])
        last_only = prefill_last_only(batch, hidden.shape[0])
        event = {
            "kind": "model_hidden", "role": role, "forward_index": calls,
            "forward_mode": batch.forward_mode.name,
            "seq_lens": _tolist(batch.seq_lens),
            "input_token_ids": _tolist(inputs[0][-1:] if last_only else inputs[0]),
            "prefill_collection": "last_row_per_chunk" if last_only else "all_rows",
            "final_hidden": summarize_hidden(hidden, positions, last_only=last_only),
            "auxiliary_hidden": [summarize_hidden(t, positions, last_only=last_only)
                                 for t in auxiliary],
            "target_capture_before_layers": list(module.layers_to_capture)
                if role == "target" else None,
            "draft_auxiliary_is_not_target_tap": role == "draft",
        }
        _write(context, event)
        calls += 1
        return None

    return hook


def create_logits_trace_hook(config: dict):
    def hook(module, inputs, output):
        context = _context(config)
        if context is None or output.next_token_logits is None:
            return None
        batch = inputs[3]
        if batch.forward_mode.is_idle():
            return None
        role = "draft" if output.next_token_logits.shape[-1] == 32000 else "target"
        _write(context, {
            "kind": "model_logits", "role": role,
            "forward_mode": batch.forward_mode.name,
            "token_id_space": "draft_32000_vocab" if role == "draft" else "target_vocab",
            "probabilities": summarize_logits(output.next_token_logits),
        })
        return None

    return hook
