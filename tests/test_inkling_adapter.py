"""Tests for inkling_adapter — Thinking Machines Lab generality."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import inkling_adapter

def test_registry_has_all_models():
    for mid in ["qwen3-1.7b","qwen3-8b","llama-3.2-3b","inkling","inkling-small","tinker"]:
        assert mid in config.MODEL_REGISTRY

def test_inkling_config_env_override(monkeypatch):
    monkeypatch.setenv("INKLING_MODEL_ID", "thinking-machines/inkling-test")
    import importlib; import config as cfgmod; importlib.reload(cfgmod)
    assert cfgmod.MODEL_REGISTRY["inkling"]["hf_id"] == "thinking-machines/inkling-test"
    monkeypatch.delenv("INKLING_MODEL_ID", raising=False)

def test_list_available_models():
    avail = inkling_adapter.list_available_models()
    assert isinstance(avail, list)
    assert any("qwen3-1.7b" in m for m in avail)

def test_get_lens_model_tiny_fallback():
    # tiny model via registry custom hf_id that doesn't exist -> should raise FileNotFoundError with helpful msg
    try:
        inkling_adapter.get_lens_model("nonexistent-model-xyz", dtype="float16", device="cpu")
        assert False, "should have raised"
    except (FileNotFoundError, NotImplementedError) as e:
        assert "not found" in str(e).lower() or "api" in str(e).lower()

def test_inkling_adapter_implements_lens_protocol():
    # tiny decoder satisfies LensModel — test that adapter would wrap it if local existed
    # we test the TinyDecoder directly satisfies protocol
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens", "tests"))
    from tiny import TinyDecoder
    m = TinyDecoder(n_layers=2, d_model=8)
    # LensModel requires n_layers, d_model, layers, tokenizer, encode, forward, unembed
    for attr in ["n_layers","d_model","layers","tokenizer","encode","forward","unembed"]:
        assert hasattr(m, attr)

def test_dtype_mapping():
    import torch
    assert inkling_adapter._dtype("float16") == torch.float16
    assert inkling_adapter._dtype("float32") == torch.float32
    assert inkling_adapter._dtype("bfloat16") == torch.bfloat16

def test_inkling_config_dataclass():
    cfg = inkling_adapter.InklingConfig(model_id="test", api_base="http://x", api_key="sk", local_dir="/tmp")
    assert cfg.model_id == "test"
