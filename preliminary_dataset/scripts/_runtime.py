"""Shared runtime helpers for loading (ref, optimized) pairs and running diff tests.

This module handles the two distinct optimized-kernel interfaces found in our sources:
  - CUDA-L1:     ``class ModelNew(nn.Module)`` (drop-in replacement for ``class Model``)
  - KernelAgent: ``def kernel_function(...)`` (free function; the benchmark.py in
                 upstream binds model parameters as kwargs)

The diff test procedure:
  1. Load ``ref.py`` and build ``Model(*get_init_inputs()).cuda().eval()``.
  2. Load ``optimized.py`` and build its counterpart.
  3. Use ``ref.get_inputs()`` to generate inputs, move to GPU, optionally convert dtype.
  4. Run both models with ``torch.inference_mode()``; compare outputs with ``torch.allclose``.
  5. Report pass/fail plus max abs diff, max rel diff, shape/dtype info.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import torch


# --------------------------------------------------------------------------
# Module loading (isolated import so a broken optimized.py cannot pollute state)
# --------------------------------------------------------------------------

def _load_module(path: Path, mod_tag: str):
    import hashlib
    name = f"_pd_{mod_tag}_{hashlib.md5(str(path).encode()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------
# Input preparation
# --------------------------------------------------------------------------

def _to_device(obj, device: torch.device, dtype: torch.dtype | None):
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        if dtype is not None and obj.is_floating_point():
            obj = obj.to(dtype)
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device, dtype) for x in obj)
    return obj


def _materialize_inputs(ref_mod, device: torch.device, dtype: torch.dtype | None):
    init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])()
    if not isinstance(init_inputs, (tuple, list)):
        init_inputs = [init_inputs]
    init_inputs = [_to_device(x, device, dtype) for x in init_inputs]

    inputs = ref_mod.get_inputs()
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]
    inputs = [_to_device(x, device, dtype) for x in inputs]
    return list(init_inputs), list(inputs)


# --------------------------------------------------------------------------
# KernelAgent adapter: bind Model parameters to kernel_function kwargs
# --------------------------------------------------------------------------

_MODEL_PARAM_KEYS = {
    "weight", "bias", "kernel_size", "stride", "padding",
    "dilation", "output_padding", "groups", "eps",
}


def _extract_model_params(model: torch.nn.Module) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for _, m in model.named_modules():
        for attr in ("weight", "bias"):
            val = getattr(m, attr, None)
            if val is not None and isinstance(val, (torch.Tensor, torch.nn.Parameter)):
                params.setdefault(attr, val)
        for attr in ("kernel_size", "stride", "padding", "dilation",
                     "output_padding", "groups", "eps"):
            val = getattr(m, attr, None)
            if val is not None:
                params.setdefault(
                    attr, val[0] if isinstance(val, (tuple, list)) else val
                )
    return params


def _bind_kernel_function(kfn: Callable, model: torch.nn.Module) -> Callable:
    """Return a callable f(*inputs) that calls kfn with model params bound."""
    sig = inspect.signature(kfn)
    param_names = list(sig.parameters.keys())
    model_params = _extract_model_params(model)
    has_var = any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in sig.parameters.values()
    )
    if not has_var and not _MODEL_PARAM_KEYS.intersection(param_names):
        return kfn
    if has_var and model_params:
        def bound_var(*args):
            return kfn(*args, **model_params)
        return bound_var

    def bound(*args):
        kwargs = {}
        pos = 0
        for name in param_names:
            if name in model_params:
                kwargs[name] = model_params[name]
            elif pos < len(args):
                kwargs[name] = args[pos]
                pos += 1
        return kfn(**kwargs)

    return bound


# --------------------------------------------------------------------------
# Diff test result
# --------------------------------------------------------------------------

@dataclass
class DiffResult:
    pair_dir: str
    interface: str                # "ModelNew" or "kernel_function"
    passed: bool = False
    error: str | None = None
    # numeric stats (None on error)
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    ref_shape: str | None = None
    opt_shape: str | None = None
    ref_dtype: str | None = None
    opt_dtype: str | None = None
    elapsed_sec: float | None = None
    rtol: float | None = None
    atol: float | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Output comparison
# --------------------------------------------------------------------------

def _normalize_outputs(out):
    """Return a list of tensors from (tensor | list[tensor] | tuple[tensor])."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return list(out)
    raise TypeError(f"unsupported output type: {type(out).__name__}")


def _compare(out_ref, out_opt, rtol: float, atol: float):
    refs = _normalize_outputs(out_ref)
    opts = _normalize_outputs(out_opt)

    if len(refs) != len(opts):
        return False, None, None, [f"output count mismatch: ref={len(refs)} opt={len(opts)}"]

    max_abs = 0.0
    max_rel = 0.0
    notes: list[str] = []
    passed = True

    for i, (r, o) in enumerate(zip(refs, opts)):
        # shape
        if r.shape != o.shape:
            # allow squeezing of trailing singleton dims as a convenience
            if r.numel() == o.numel():
                notes.append(f"out[{i}] shape differs but numel matches: {tuple(r.shape)} vs {tuple(o.shape)}")
                o = o.reshape(r.shape)
            else:
                passed = False
                notes.append(f"out[{i}] shape mismatch: ref={tuple(r.shape)} opt={tuple(o.shape)}")
                continue

        # dtype
        if r.dtype != o.dtype:
            notes.append(f"out[{i}] dtype differs: ref={r.dtype} opt={o.dtype}; casting for comparison")
            o = o.to(r.dtype)

        # compare on cpu-fp32 for numerical stability
        r_cmp = r.detach().float().cpu()
        o_cmp = o.detach().float().cpu()
        abs_diff = (r_cmp - o_cmp).abs()
        max_abs = max(max_abs, float(abs_diff.max().item()))
        rel = abs_diff / (r_cmp.abs().clamp(min=1e-8))
        max_rel = max(max_rel, float(rel.max().item()))

        if not torch.allclose(r_cmp, o_cmp, rtol=rtol, atol=atol, equal_nan=False):
            passed = False
            notes.append(f"out[{i}] allclose failed (rtol={rtol}, atol={atol})")

    return passed, max_abs, max_rel, notes


# --------------------------------------------------------------------------
# The core diff tester
# --------------------------------------------------------------------------

def run_diff_test(
    pair_dir: Path,
    *,
    device: str = "cuda",
    dtype: str | None = None,  # "bf16", "fp16", "fp32", or None (keep as-is)
    rtol: float = 1e-3,
    atol: float = 1e-3,
    seed: int = 42,
) -> DiffResult:
    """Load (ref.py, optimized.py) from *pair_dir* and compare outputs."""
    result = DiffResult(pair_dir=str(pair_dir), interface="?")
    result.rtol = rtol
    result.atol = atol

    torch_dtype = {
        None: None,
        "fp32": torch.float32, "float32": torch.float32,
        "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    }.get(dtype, None if dtype is None else getattr(torch, dtype, None))

    dev = torch.device(device)

    # Determine interface from meta.json (fallback: inspect optimized.py)
    meta_path = pair_dir / "meta.json"
    interface = "ModelNew"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            interface = meta.get("optimized_interface", interface)
        except Exception:
            pass
    result.interface = interface

    ref_path = pair_dir / "ref.py"
    opt_path = pair_dir / "optimized.py"
    if not ref_path.exists():
        result.error = f"missing ref.py at {ref_path}"
        return result
    if not opt_path.exists():
        result.error = f"missing optimized.py at {opt_path}"
        return result

    t0 = time.perf_counter()
    try:
        ref_mod = _load_module(ref_path, "ref")
    except Exception as e:
        result.error = f"failed to import ref.py: {e}"
        result.notes.append(traceback.format_exc())
        return result

    try:
        opt_mod = _load_module(opt_path, "opt")
    except Exception as e:
        result.error = f"failed to import optimized.py: {e}"
        result.notes.append(traceback.format_exc())
        return result

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        init_inputs, inputs = _materialize_inputs(ref_mod, dev, torch_dtype)
    except Exception as e:
        result.error = f"failed to materialize inputs: {e}"
        result.notes.append(traceback.format_exc())
        return result

    try:
        RefModel = ref_mod.Model
    except AttributeError:
        result.error = "ref.py has no `Model` class"
        return result

    try:
        torch.manual_seed(seed)
        ref_model = RefModel(*init_inputs).to(dev).eval()
        if torch_dtype is not None and any(p.numel() > 0 for p in ref_model.parameters()):
            ref_model = ref_model.to(torch_dtype)
    except Exception as e:
        result.error = f"failed to build ref Model: {e}"
        result.notes.append(traceback.format_exc())
        return result

    # Build opt side
    if interface == "ModelNew":
        if not hasattr(opt_mod, "ModelNew"):
            result.error = "optimized.py has no `ModelNew` class (expected for CUDA-L1-style)"
            return result
        try:
            torch.manual_seed(seed)
            opt_model = opt_mod.ModelNew(*init_inputs).to(dev).eval()
            if torch_dtype is not None and any(p.numel() > 0 for p in opt_model.parameters()):
                opt_model = opt_model.to(torch_dtype)
            # Copy ref weights into opt to ensure identical parameters
            ref_state = ref_model.state_dict()
            if ref_state:
                try:
                    opt_model.load_state_dict(ref_state, strict=False)
                except Exception as e:
                    result.notes.append(f"load_state_dict(strict=False) note: {e}")
            opt_call = lambda *args: opt_model(*args)
        except Exception as e:
            result.error = f"failed to build ModelNew: {e}"
            result.notes.append(traceback.format_exc())
            return result

    elif interface == "kernel_function":
        if not hasattr(opt_mod, "kernel_function"):
            result.error = "optimized.py has no `kernel_function` (expected for KernelAgent-style)"
            return result
        try:
            opt_call = _bind_kernel_function(opt_mod.kernel_function, ref_model)
        except Exception as e:
            result.error = f"failed to bind kernel_function: {e}"
            result.notes.append(traceback.format_exc())
            return result
    else:
        result.error = f"unknown interface: {interface}"
        return result

    # Run both
    try:
        with torch.inference_mode():
            out_ref = ref_model(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            out_opt = opt_call(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    except Exception as e:
        result.error = f"runtime error during forward: {e}"
        result.notes.append(traceback.format_exc())
        return result

    # Record shape/dtype
    try:
        refs = _normalize_outputs(out_ref)
        opts = _normalize_outputs(out_opt)
        result.ref_shape = "|".join(str(tuple(r.shape)) for r in refs)
        result.opt_shape = "|".join(str(tuple(o.shape)) for o in opts)
        result.ref_dtype = "|".join(str(r.dtype) for r in refs)
        result.opt_dtype = "|".join(str(o.dtype) for o in opts)
    except Exception as e:
        result.notes.append(f"failed to record shape/dtype: {e}")

    passed, max_abs, max_rel, notes = _compare(out_ref, out_opt, rtol=rtol, atol=atol)
    result.passed = passed
    result.max_abs_diff = max_abs
    result.max_rel_diff = max_rel
    result.notes.extend(notes)
    result.elapsed_sec = time.perf_counter() - t0
    return result


def result_to_dict(r: DiffResult) -> dict:
    return asdict(r)
