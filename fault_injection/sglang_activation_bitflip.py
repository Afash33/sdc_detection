"""One-shot BF16 activation bit flip for SGLang forward hooks.

This module is imported inside an SGLang model worker through the
``--forward-hooks`` option.  The hook remains dormant until ``arm_path``
exists, mutates one activation element in place, records evidence in JSON,
removes the arm file, and never fires again in that worker process.

The initial implementation intentionally targets activation outputs only.
It does not claim to emulate an instruction-level CUDA fault or a weight
fault that must occur before a module forward.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


def _parse_bit_positions(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        positions = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        positions = tuple(int(item) for item in value)
    else:
        raise TypeError("bit_positions must be an integer or a sequence of integers")

    if len(positions) not in (1, 2):
        raise ValueError("bit_positions must contain exactly one or two positions")
    if len(set(positions)) != len(positions):
        raise ValueError("bit_positions must not contain duplicates")
    if any(position < 0 or position > 15 for position in positions):
        raise ValueError("BF16 bit positions must be in the inclusive range [0, 15]")
    return positions


def _activation_tensor(output: Any) -> tuple[torch.Tensor, str]:
    if isinstance(output, torch.Tensor):
        return output, "tensor"
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], "tuple[0]"
    raise TypeError(
        "activation bit-flip hook expected a Tensor or a tuple whose first item is a Tensor"
    )


def _signed_int16(unsigned_value: int) -> int:
    return unsigned_value if unsigned_value < 0x8000 else unsigned_value - 0x10000


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def create_activation_bitflip_hook(config: Mapping[str, Any]) -> Callable[..., Any]:
    """Create an armable, one-shot SGLang forward hook.

    Required configuration:
      - ``arm_path``: injection occurs on the first matched forward after this
        file appears.
      - ``event_path``: JSON evidence written after a successful injection.
      - ``bit_positions``: one or two BF16 bit positions in ``[0, 15]``.

    Optional configuration:
      - ``flat_index``: activation element index after flattening (default 0;
        negative indexing is supported).
      - ``fault_id`` and ``module_name``: labels copied into the event record.
    """

    if not isinstance(config, Mapping):
        raise TypeError("hook config must be a mapping")

    arm_path = Path(str(config.get("arm_path", "")))
    event_path = Path(str(config.get("event_path", "")))
    if not str(config.get("arm_path", "")):
        raise ValueError("arm_path is required")
    if not str(config.get("event_path", "")):
        raise ValueError("event_path is required")

    bit_positions = _parse_bit_positions(config.get("bit_positions"))
    flat_index = int(config.get("flat_index", 0))
    fault_id = str(config.get("fault_id", "target_activation_bitflip"))
    configured_module_name = config.get("module_name")
    mask = sum(1 << position for position in bit_positions)

    state = {"call_count": 0, "triggered": False}

    def hook(module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        state["call_count"] += 1
        if state["triggered"] or not arm_path.is_file():
            return output

        tensor, output_location = _activation_tensor(output)
        if tensor.dtype != torch.bfloat16:
            raise TypeError(
                f"activation bit-flip hook requires torch.bfloat16, got {tensor.dtype}"
            )
        if tensor.numel() == 0:
            raise ValueError("cannot inject into an empty activation tensor")
        if not tensor.is_contiguous():
            raise ValueError(
                "activation tensor must be contiguous so the bit view mutates the original output"
            )

        normalized_index = flat_index
        if normalized_index < 0:
            normalized_index += tensor.numel()
        if normalized_index < 0 or normalized_index >= tensor.numel():
            raise IndexError(
                f"flat_index {flat_index} is outside activation with {tensor.numel()} elements"
            )

        flat_tensor = tensor.reshape(-1)
        flat_bits = tensor.view(torch.int16).reshape(-1)
        before_value = float(flat_tensor[normalized_index].item())
        before_bits = int(flat_bits[normalized_index].item()) & 0xFFFF
        after_bits = before_bits ^ mask

        with torch.no_grad():
            flat_bits[normalized_index] = _signed_int16(after_bits)

        after_value = float(flat_tensor[normalized_index].item())
        observed_after_bits = int(flat_bits[normalized_index].item()) & 0xFFFF
        if observed_after_bits != after_bits:
            raise RuntimeError(
                f"bit flip verification failed: expected 0x{after_bits:04x}, "
                f"observed 0x{observed_after_bits:04x}"
            )

        event = {
            "status": "injected",
            "fault_id": fault_id,
            "fault_kind": "activation_bitflip",
            "fault_mode": "single" if len(bit_positions) == 1 else "double",
            "module_name": configured_module_name,
            "module_type": type(module).__name__,
            "worker_pid": os.getpid(),
            "timestamp_unix": time.time(),
            "call_index": state["call_count"],
            "output_location": output_location,
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            "tensor_device": str(tensor.device),
            "flat_index_requested": flat_index,
            "flat_index_normalized": normalized_index,
            "bit_positions": list(bit_positions),
            "xor_mask_hex": f"0x{mask:04x}",
            "before_value": before_value,
            "after_value": after_value,
            "before_bits_hex": f"0x{before_bits:04x}",
            "after_bits_hex": f"0x{after_bits:04x}",
            "arm_path": str(arm_path),
            "event_path": str(event_path),
        }

        _atomic_write_json(event_path, event)
        try:
            arm_path.unlink()
        except FileNotFoundError:
            pass
        state["triggered"] = True
        return output

    return hook
