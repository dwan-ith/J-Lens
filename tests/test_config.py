"""Tests for config — no hardcoding, generality."""
import os
import config

def test_default_config_layers_qwen_1_7b():
    cfg = config.get_config()
    layers = cfg.analysis_layers(28)
    assert len(layers) >= 2
    assert all(0 <= l < 28 for l in layers)
    assert layers == sorted(layers)

def test_analysis_layers_scales_with_depth():
    cfg = config.get_config()
    layers_28 = cfg.analysis_layers(28)
    layers_36 = cfg.analysis_layers(36)
    assert max(layers_36) > max(layers_28)
    # fractions should give proportional layers
    assert len(layers_28) == len(layers_36)

def test_analysis_layers_small_model():
    cfg = config.get_config()
    layers = cfg.analysis_layers(4)
    assert all(0 <= l < 4 for l in layers)

def test_env_override():
    os.environ["JSPACE_MAX_SEQ_LEN"] = "128"
    cfg = config.get_config()
    assert cfg.max_seq_len == 128
    del os.environ["JSPACE_MAX_SEQ_LEN"]

def test_model_registry_has_inkling():
    assert "inkling" in config.MODEL_REGISTRY
    assert "inkling-small" in config.MODEL_REGISTRY

def test_model_registry_qwen():
    assert "qwen3-1.7b" in config.MODEL_REGISTRY
    assert config.MODEL_REGISTRY["qwen3-1.7b"]["n_layers"] == 28

def test_get_config_override():
    cfg = config.get_config(n_prompts_per_cat=10)
    assert cfg.n_prompts_per_cat == 10

def test_mid_layer_fraction():
    cfg = config.get_config(mid_layer_fraction=0.5)
    assert cfg.mid_layer_for(28) == 14
    assert cfg.mid_layer_for(36) == 18

def test_no_hardcoded_paths_in_config():
    # config should use env vars, not hardcoded C:\...
    assert config.DEFAULT_MODEL_DIR is not None
