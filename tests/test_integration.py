"""Integration tests with tiny model — no large weights needed."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens", "tests"))
import torch
import numpy as np
from jlens.hooks import ActivationRecorder
from tiny import TinyDecoder as Tiny2
import jlens
import analysis_utils as au
import config

def _tiny():
    return Tiny2(n_layers=4, d_model=8, vocab_size=32, seed=0)

def test_tiny_model_forward():
    m = _tiny()
    ids = m.encode("hello world", max_length=32)
    out = m.forward(ids)
    assert hasattr(out, "last_hidden_state")
    assert out.last_hidden_state.shape[-1] == 8

def test_tiny_jacobian_fit():
    m = _tiny()
    prompts = ["hello world this is a test prompt for jacobian fitting with enough length", "another prompt with different text to average over for the lens fitting"]
    lens = jlens.fit(m, prompts, source_layers=[0,1,2], target_layer=3, max_seq_len=32)
    assert 0 in lens.jacobians
    assert lens.jacobians[0].shape == (8,8)

def test_tiny_lens_apply():
    m = _tiny()
    prompts = ["test prompt " * 10]
    lens = jlens.fit(m, prompts, source_layers=[0,1], target_layer=3, max_seq_len=32)
    logits, _, _ = lens.apply(m, "hello world test", layers=[0,1], positions=[-1])
    assert 0 in logits
    assert logits[0].shape[-1] == 32

def test_activation_recorder_tiny():
    m = _tiny()
    ids = m.encode("hello world test prompt", max_length=32)
    with ActivationRecorder(m.layers, at=[0,2]) as rec:
        m.forward(ids)
        assert 0 in rec.activations
        assert 2 in rec.activations
        assert rec.activations[0].shape[-1] == 8
        assert rec.activations[2].shape[-1] == 8

def test_measure_all_layers_tiny():
    import safety_gap as sg
    orig_layers = sg.ANALYSIS_LAYERS
    sg.ANALYSIS_LAYERS = [0,1,2]
    m = _tiny()
    class Tok:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=32):
            from types import SimpleNamespace
            return SimpleNamespace(input_ids=m.encode(text, max_length=max_length))
        def __getattr__(self, name):
            return getattr(m.tokenizer, name)
    tok = Tok()
    fake_lens = type("Fake", (), {"jacobians": {0: torch.eye(8), 1: torch.eye(8), 2: torch.eye(8)}})()
    try:
        res = sg.measure_all_layers(m, tok, "hello world test prompt with enough length for tiny model", fake_lens)
        assert 0 in res and 1 in res and 2 in res
        for layer in [0,1,2]:
            assert "jl_cos" in res[layer] and "ll_cos" in res[layer] and "rand_cos" in res[layer]
            assert -1 <= res[layer]["jl_cos"] <= 1
    finally:
        sg.ANALYSIS_LAYERS = orig_layers

def test_safety_gap_bootstrap_on_tiny():
    # test that bootstrap works on tiny pooled scores
    in_scores = [0.5, 0.6, 0.55]
    safety_scores = [0.7, 0.72, 0.68]
    gap, lo, hi, p = au.bootstrap_diff_ci(safety_scores, in_scores, n_boot=500, seed=0)
    assert gap > 0
    assert lo <= gap <= hi

def test_deception_trajectory_tiny():
    """Test that trajectory extraction works on tiny model (not deception detection itself)."""
    m = _tiny()
    lens_prompts = ["hello world " * 8, "test prompt " * 8]
    lens = jlens.fit(m, lens_prompts, source_layers=[0,1,2], target_layer=3, max_seq_len=32)
    # get trajectory on tiny (4 layers)
    ids = m.encode("hello world deceptive test", max_length=32)
    W_U = m.lm_head.weight.data.cpu().float()
    with torch.no_grad():
        with ActivationRecorder(m.layers, at=[0,1,2]) as rec:
            out = m.forward(ids)
            # tiny forward returns SimpleNamespace, not HF logits — so we test unembed path
            h = rec.activations[0][0,-1,:].float()
            logits = m.unembed(h.unsqueeze(0)).float()
            assert logits.shape[-1] == 32

def test_config_generalizes_to_tiny():
    cfg = config.get_config()
    layers = cfg.analysis_layers(4)
    assert all(0 <= l < 4 for l in layers)
    assert len(layers) >= 2

def test_no_hardcoded_analysis_layers_import():
    import safety_gap, deception_detector, cross_arch
    import inspect
    # each should have ANALYSIS_LAYERS derived from config, not literal [4,9,14,19,24] only
    for mod in [safety_gap, deception_detector, cross_arch]:
        src = inspect.getsource(mod)
        assert "config" in src, f"{mod.__name__} does not import config"

def test_trajectory_features_key_contract():
    """Regression test for KeyError: 'jl_L14' — every feats[...] access in
    deception_detector.main must exist in trajectory_features() output."""
    import re, pathlib
    import deception_detector as dd
    feats = dd.trajectory_features([0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.1, 0.2])
    src = pathlib.Path(dd.__file__).read_text()
    accessed = set(re.findall(r'feats\["(\w+)"\]', src))
    for key in accessed:
        assert key in feats, f"deception_detector.main accesses feats[{key!r}] but trajectory_features() does not return it"

def test_resolve_blocks_tiny():
    """_resolve_blocks must handle TinyDecoder layout (model.layers, no .model)."""
    import deception_detector as dd
    m = _tiny()
    text_module, block_layers = dd._resolve_blocks(m)
    assert block_layers is m.layers
    assert text_module is m
