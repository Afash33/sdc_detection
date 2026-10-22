"""Controlled BF16 weight bit flips inside an SGLang model worker.

SGLang keeps model parameters in a worker process, so an HTTP client cannot
mutate ``module.weight`` directly.  This module uses SGLang's existing
``update_weights_from_tensor`` custom-loader path as a control channel.  The
loader runs inside the Target worker, performs an in-place XOR, records exact
bit evidence, and retains the original bits until an explicit restore command.

The initial implementation deliberately supports one TP=1 Target fault at a
time.  Draft, tensor-parallel, quantized, and non-BF16 weights require separate
coordinate and storage semantics and are therefore rejected or left for later
work.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


LOADER_PATH = "fault_injection.sglang_weight_bitflip.sdc_weight_fault_loader"
_CONTROL_PREFIX = "__sdc_weight_fault_control_v1__:"
_STATE_ATTRIBUTE = "_sdc_weight_fault_state_v1"


class _ByValueControlTensor:
    """Pickle as a tiny Tensor value instead of a shared-memory Tensor handle."""

    def __reduce__(self):
        return torch.tensor, ([0],)


def _parse_bit_positions(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        positions = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        positions = tuple(int(item) for item in value)
    else:
        raise TypeError("bit_positions must be an integer or a sequence")
    if len(positions) not in (1, 2):
        raise ValueError("bit_positions must contain exactly one or two positions")
    if len(set(positions)) != len(positions):
        raise ValueError("bit_positions must not contain duplicates")
    if any(position < 0 or position > 15 for position in positions):
        raise ValueError("BF16 bit positions must be in the inclusive range [0, 15]")
    return positions


def _signed_int16(unsigned_value: int) -> int:
    return unsigned_value if unsigned_value < 0x8000 else unsigned_value - 0x10000


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def encode_control_name(command: Mapping[str, Any]) -> str:
    """Encode a JSON command in the synthetic named-tensor name."""

    raw = json.dumps(dict(command), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    return _CONTROL_PREFIX + encoded


def decode_control_name(name: str) -> dict[str, Any]:
    """Decode and validate a synthetic named-tensor control name."""

    if not name.startswith(_CONTROL_PREFIX):
        raise ValueError("custom loader received a non-SDC control tensor")
    encoded = name[len(_CONTROL_PREFIX) :]
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        command = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid SDC weight-fault control payload") from error
    if not isinstance(command, dict):
        raise TypeError("SDC weight-fault command must decode to a JSON object")
    return command


def _one_command(named_tensors: Any) -> dict[str, Any]:
    items = list(named_tensors)
    if len(items) != 1 or len(items[0]) != 2:
        raise ValueError("SDC weight loader requires exactly one control tensor")
    name, _unused_tensor = items[0]
    return decode_control_name(str(name))


def _resolve_weight(
    model: torch.nn.Module, module_name: str, parameter_name: str
) -> tuple[torch.nn.Module, torch.nn.Parameter]:
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"model has no module named {module_name!r}")
    module = modules[module_name]
    parameter = getattr(module, parameter_name, None)
    if not isinstance(parameter, torch.nn.Parameter):
        raise TypeError(
            f"{module_name}.{parameter_name} is not a torch.nn.Parameter"
        )
    if parameter.dtype != torch.bfloat16:
        raise TypeError(
            f"weight bit flip requires torch.bfloat16, got {parameter.dtype}"
        )
    if parameter.numel() == 0:
        raise ValueError("cannot inject into an empty parameter")
    if not parameter.is_contiguous():
        raise ValueError("parameter must be contiguous for an in-place bit view")
    return module, parameter


def _normalize_index(flat_index: int, numel: int) -> int:
    normalized = flat_index + numel if flat_index < 0 else flat_index
    if normalized < 0 or normalized >= numel:
        raise IndexError(f"flat_index {flat_index} is outside weight with {numel} elements")
    return normalized


def _model_state(model: torch.nn.Module) -> dict[str, Any]:
    state = getattr(model, _STATE_ATTRIBUTE, None)
    if state is None:
        state = {"active": None}
        setattr(model, _STATE_ATTRIBUTE, state)
    return state


def _inject(model: torch.nn.Module, command: Mapping[str, Any]) -> None:
    state = _model_state(model)
    if state["active"] is not None:
        active_id = state["active"]["fault_id"]
        raise RuntimeError(f"weight fault {active_id!r} is still active; restore it first")

    fault_id = str(command.get("fault_id", ""))
    module_name = str(command.get("module_name", ""))
    parameter_name = str(command.get("parameter_name", "weight"))
    event_path_value = str(command.get("event_path", ""))
    if not fault_id:
        raise ValueError("fault_id is required")
    if not module_name:
        raise ValueError("module_name is required")
    if not event_path_value:
        raise ValueError("event_path is required")

    bit_positions = _parse_bit_positions(command.get("bit_positions"))
    flat_index = int(command.get("flat_index", 0))
    module, parameter = _resolve_weight(model, module_name, parameter_name)
    normalized_index = _normalize_index(flat_index, parameter.numel())
    flat_values = parameter.reshape(-1)
    flat_bits = parameter.view(torch.int16).reshape(-1)
    before_value = float(flat_values[normalized_index].item())
    before_bits = int(flat_bits[normalized_index].item()) & 0xFFFF
    mask = sum(1 << position for position in bit_positions)
    after_bits = before_bits ^ mask

    with torch.no_grad():
        flat_bits[normalized_index] = _signed_int16(after_bits)

    observed_bits = int(flat_bits[normalized_index].item()) & 0xFFFF
    after_value = float(flat_values[normalized_index].item())
    if observed_bits != after_bits:
        raise RuntimeError(
            f"weight flip verification failed: expected 0x{after_bits:04x}, "
            f"observed 0x{observed_bits:04x}"
        )

    active = {
        "fault_id": fault_id,
        "module_name": module_name,
        "parameter_name": parameter_name,
        "flat_index_requested": flat_index,
        "flat_index_normalized": normalized_index,
        "bit_positions": list(bit_positions),
        "xor_mask": mask,
        "before_bits": before_bits,
        "after_bits": after_bits,
        "injected_event_path": event_path_value,
    }
    state["active"] = active
    event = {
        "status": "injected",
        "action": "inject",
        "fault_kind": "weight_bitflip",
        "fault_mode": "single" if len(bit_positions) == 1 else "double",
        "fault_id": fault_id,
        "module_name": module_name,
        "module_type": type(module).__name__,
        "parameter_name": parameter_name,
        "parameter_shape": list(parameter.shape),
        "parameter_dtype": str(parameter.dtype),
        "parameter_device": str(parameter.device),
        "flat_index_requested": flat_index,
        "flat_index_normalized": normalized_index,
        "bit_positions": list(bit_positions),
        "xor_mask_hex": f"0x{mask:04x}",
        "before_value": before_value,
        "after_value": after_value,
        "before_bits_hex": f"0x{before_bits:04x}",
        "after_bits_hex": f"0x{after_bits:04x}",
        "worker_pid": os.getpid(),
        "timestamp_unix": time.time(),
        "event_path": event_path_value,
    }
    _atomic_write_json(Path(event_path_value), event)


def _restore(model: torch.nn.Module, command: Mapping[str, Any]) -> None:
    state = _model_state(model)
    active = state["active"]
    if active is None:
        raise RuntimeError("there is no active weight fault to restore")

    requested_fault_id = str(command.get("fault_id", ""))
    event_path_value = str(command.get("event_path", ""))
    if requested_fault_id != active["fault_id"]:
        raise ValueError(
            f"restore fault_id {requested_fault_id!r} does not match active "
            f"fault {active['fault_id']!r}"
        )
    if not event_path_value:
        raise ValueError("event_path is required")

    module, parameter = _resolve_weight(
        model, active["module_name"], active["parameter_name"]
    )
    index = active["flat_index_normalized"]
    flat_values = parameter.reshape(-1)
    flat_bits = parameter.view(torch.int16).reshape(-1)
    current_bits = int(flat_bits[index].item()) & 0xFFFF
    current_value = float(flat_values[index].item())
    if current_bits != active["after_bits"]:
        raise RuntimeError(
            f"refusing restore because active weight changed: expected fault bits "
            f"0x{active['after_bits']:04x}, observed 0x{current_bits:04x}"
        )

    with torch.no_grad():
        flat_bits[index] = _signed_int16(active["before_bits"])

    restored_bits = int(flat_bits[index].item()) & 0xFFFF
    restored_value = float(flat_values[index].item())
    if restored_bits != active["before_bits"]:
        raise RuntimeError(
            f"weight restore verification failed: expected 0x{active['before_bits']:04x}, "
            f"observed 0x{restored_bits:04x}"
        )

    event = {
        "status": "restored",
        "action": "restore",
        "fault_kind": "weight_bitflip",
        "fault_id": active["fault_id"],
        "module_name": active["module_name"],
        "module_type": type(module).__name__,
        "parameter_name": active["parameter_name"],
        "parameter_shape": list(parameter.shape),
        "parameter_dtype": str(parameter.dtype),
        "parameter_device": str(parameter.device),
        "flat_index_requested": active["flat_index_requested"],
        "flat_index_normalized": index,
        "bit_positions": active["bit_positions"],
        "xor_mask_hex": f"0x{active['xor_mask']:04x}",
        "fault_bits_hex": f"0x{current_bits:04x}",
        "restored_bits_hex": f"0x{restored_bits:04x}",
        "fault_value": current_value,
        "restored_value": restored_value,
        "injected_event_path": active["injected_event_path"],
        "worker_pid": os.getpid(),
        "timestamp_unix": time.time(),
        "event_path": event_path_value,
    }
    _atomic_write_json(Path(event_path_value), event)
    state["active"] = None


def sdc_weight_fault_loader(model: torch.nn.Module, named_tensors: Any) -> None:
    """SGLang custom weight loader used by ``update_weights_from_tensor``."""

    command = _one_command(named_tensors)
    action = command.get("action")
    if action == "inject":
        _inject(model, command)
    elif action == "restore":
        _restore(model, command)
    else:
        raise ValueError(f"unsupported weight-fault action: {action!r}")


def build_update_payload(command: Mapping[str, Any], tp_size: int = 1) -> dict[str, Any]:
    """Build the JSON body for SGLang's ``update_weights_from_tensor`` endpoint."""

    if tp_size != 1:
        raise ValueError("the initial weight fault controller supports only tp_size=1")

    # SGLang's helper uses ForkingPickler's shared-memory reduction for ordinary
    # CPU tensors.  That requires the HTTP client and server to share a
    # multiprocessing auth key, which independent CLI processes do not.  The
    # proxy below reconstructs a harmless one-element Tensor by value.  SGLang
    # can still call ``.to(device)`` on it, while the actual command remains in
    # the synthetic tensor name.
    named_tensors = [(encode_control_name(command), _ByValueControlTensor())]
    raw = pickle.dumps(named_tensors, protocol=pickle.HIGHEST_PROTOCOL)
    serialized = base64.b64encode(raw).decode("ascii")
    return {
        "serialized_named_tensors": [serialized],
        "load_format": LOADER_PATH,
        "flush_cache": True,
        "abort_all_requests": True,
        "disable_draft_model": True,
        "torch_empty_cache": False,
    }


def send_update(base_url: str, command: Mapping[str, Any], timeout: float = 30.0) -> dict:
    """Send one synchronous Target-worker fault command to an SGLang server."""

    payload = build_update_payload(command)
    url = base_url.rstrip("/") + "/update_weights_from_tensor"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang weight-fault request failed ({error.code}): {body}") from error
    if not result.get("success"):
        raise RuntimeError(f"SGLang rejected weight-fault command: {result}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject or restore one BF16 Target weight fault in SGLang."
    )
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    subparsers = parser.add_subparsers(dest="action", required=True)

    inject = subparsers.add_parser("inject")
    inject.add_argument("--fault-id", required=True)
    inject.add_argument("--module-name", required=True)
    inject.add_argument("--parameter-name", default="weight")
    inject.add_argument("--flat-index", type=int, required=True)
    inject.add_argument(
        "--bit-position", dest="bit_positions", type=int, action="append", required=True
    )
    inject.add_argument("--event-path", required=True)

    restore = subparsers.add_parser("restore")
    restore.add_argument("--fault-id", required=True)
    restore.add_argument("--event-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = {
        key: value
        for key, value in vars(args).items()
        if key not in {"url"} and value is not None
    }
    if args.action == "inject":
        _parse_bit_positions(args.bit_positions)
    try:
        result = send_update(args.url, command)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
