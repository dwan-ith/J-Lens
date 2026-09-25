"""
Inkling/Tinker adapter — unified interface for Thinking Machines Lab models.

Abstracts over:
  - local HF weights (Qwen, Llama)
  - Inkling/Inkling-Small/Tinker API (when provided)
  - fallback to mock for testing

All safety experiments use this adapter, not direct transformers calls,
so the same code runs on open and closed models (generality).

Usage:
  from inkling_adapter import get_lens_model, InklingConfig
  model = get_lens_model("inkling")  # or "qwen3-1.7b", "tinker"
  # model satisfies jlens.protocol.LensModel (n_layers, d_model, layers, tokenizer, encode, forward, unembed)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from config import MODEL_REGISTRY, get_config

@dataclass
class InklingConfig:
    model_id: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    local_dir: Optional[str] = None

def _resolve_inkling_config(model_id: str) -> InklingConfig:
    info = MODEL_REGISTRY.get(model_id, {})
    return InklingConfig(
        model_id=info.get("hf_id", model_id),
        api_base=info.get("api_base") or os.environ.get("INKLING_API_BASE"),
        api_key=os.environ.get(info.get("api_key_env", "INKLING_API_KEY")) if info.get("api_key_env") else os.environ.get("INKLING_API_KEY"),
        local_dir=info.get("local_dir"),
    )

def get_lens_model(model_id: str = "qwen3-1.7b", dtype: str = "float16", device: str = "cpu"):
    """Return a LensModel for any registry entry. Handles local, HF hub, and API."""
    orig_id = model_id
    cfg = get_config(model_id=model_id, dtype=dtype, device=device)
    info = MODEL_REGISTRY.get(model_id)
    if info is None:
        # treat model_id as HF id directly (keep orig_id for error messages)
        info = {"hf_id": model_id, "local_dir": None}

    # 0) Mock mode first — CI must never pay a network download to reach it
    if os.environ.get("JSPACE_MOCK_INKLING") == "1":
        # For CI/testing without real Inkling weights: return TinyDecoder mock
        _tiny_tests_dir = Path(__file__).parent / "jacobian-lens" / "tests"
        if not _tiny_tests_dir.exists():
            raise ImportError(f"mock TinyDecoder dir missing: {_tiny_tests_dir}")
        if str(_tiny_tests_dir) not in sys.path:
            sys.path.insert(0, str(_tiny_tests_dir))
        try:
            from tiny import TinyDecoder
        except ImportError as e:
            raise ImportError(f"cannot import TinyDecoder for mock mode: {e}")
        print(f"[inkling_adapter] MOCK mode for {orig_id} — returning TinyDecoder")
        return TinyDecoder(n_layers=4, d_model=8, vocab_size=32)

    # 1) Try local dir (explicit cfg override wins over registry)
    local_dir = cfg.local_dir or info.get("local_dir")
    if local_dir and Path(local_dir).exists():
        import transformers
        from jlens.hf import from_hf
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            local_dir, torch_dtype=_dtype(dtype), device_map="auto",
            low_cpu_mem_usage=True, local_files_only=True,
        )
        tok = transformers.AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        return from_hf(hf, tok)

    # 2) Try HF hub
    hf_id = info.get("hf_id")
    if hf_id:
        try:
            import transformers
            from jlens.hf import from_hf
            hf = transformers.AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=_dtype(dtype), device_map="auto",
                low_cpu_mem_usage=True,
            )
            tok = transformers.AutoTokenizer.from_pretrained(hf_id)
            return from_hf(hf, tok)
        except Exception as e:
            print(f"[inkling_adapter] HF load failed for {hf_id}: {e}")

    # 3) API mode (Inkling/Tinker) — requires API base/key
    inkling_cfg = _resolve_inkling_config(orig_id)
    if inkling_cfg.api_base and inkling_cfg.api_key:
        # Thinking Machines Lab provides Inkling via API (not local weights).
        # The LensModel protocol requires local forward/unembed, so we wrap
        # the remote API: encode locally, forward via API, unembed via API.
        # This is the correct integration point for the grant — the pilot
        # falls back to open models, but the adapter is ready for API.
        # See docs/inkling_api.md for the planned RemoteLensModel.
        raise NotImplementedError(
            f"Inkling API mode for {orig_id} at {inkling_cfg.api_base}: "
            "grant provides API access — implement RemoteLensModel wrapper "
            "(see inkling_adapter.RemoteLensModel stub). Fallback to open model for pilot."
        )

    raise FileNotFoundError(f"Model {orig_id} not found locally nor on HF hub, and no API configured. Set JSPACE_MODEL_DIR or INKLING_API_BASE.")

def _dtype(dtype_str: str):
    try:
        return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype_str!r} — expected one of float16/float32/bfloat16")

def list_available_models():
    """List registry models that are resolvable (local exists or hf_id)."""
    available = []
    for mid, info in MODEL_REGISTRY.items():
        local = info.get("local_dir")
        if local and Path(local).exists():
            available.append(mid)
        elif info.get("hf_id"):
            available.append(mid + " (hf)")
    return available


class RemoteLensModel:
    """Stub for Thinking Machines API-backed LensModel.

    When the grant provides Inkling via API (no local weights), this class
    wraps the remote endpoint to satisfy `jlens.protocol.LensModel`:
      - `encode` is local (tokenizer)
      - `forward` and `unembed` are remote calls (batched, autograd not needed
        for readout; for `jacobian_for_prompt` we use the `.forward` that builds
        a graph — API mode will use finite-difference or provided Jacobian endpoint).

    Pilot uses local Qwen; this stub is here so Phase 3 on Inkling is
    implementable without changing experiment code (generality).
    """
    def __init__(self, model_id: str, api_base: str, api_key: str):
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        # TODO(phase3): implement when grant provides API spec
        raise NotImplementedError(
            f"RemoteLensModel({model_id}): implement with Inkling API spec. "
            "Set JSPACE_MOCK_INKLING=1 for testing, or use qwen3-1.7b for the pilot."
        )
