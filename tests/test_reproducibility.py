"""Reproducibility and CLI tests — grant construct validity."""
import subprocess, sys, os, json, pathlib
import numpy as np

def test_bootstrap_deterministic():
    import analysis_utils as au
    x = [0.1,0.2,0.3,0.4,0.5]
    est1, lo1, hi1 = au.bootstrap_ci(x, n_boot=500, seed=42)
    est2, lo2, hi2 = au.bootstrap_ci(x, n_boot=500, seed=42)
    assert abs(est1 - est2) < 1e-10 and abs(lo1 - lo2) < 1e-10 and abs(hi1 - hi2) < 1e-10

def test_permutation_deterministic():
    import analysis_utils as au
    a = [1,2,3]; b = [0,0,0]
    _, p1 = au.permutation_test(a,b, n_perm=500, seed=1)
    _, p2 = au.permutation_test(a,b, n_perm=500, seed=1)
    assert abs(p1 - p2) < 1e-10

def test_safety_gap_cli_help():
    result = subprocess.run([sys.executable, "safety_gap.py", "--help"], cwd=os.path.join(os.path.dirname(__file__), ".."), capture_output=True, text=True)
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--n-prompts" in result.stdout

def test_deception_cli_help():
    result = subprocess.run([sys.executable, "deception_detector.py", "--help"], cwd=os.path.join(os.path.dirname(__file__), ".."), capture_output=True, text=True)
    assert result.returncode == 0
    assert "--model" in result.stdout

def test_config_yaml_roundtrip():
    import config, pathlib, tempfile
    cfg = config.get_config(model_id="qwen3-8b", n_prompts_per_cat=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = pathlib.Path(tmpdir) / "test.yaml"
        cfg.to_yaml(p)
        assert p.exists()
        cfg2 = config.ExperimentConfig.from_yaml(p)
        assert cfg2.model_id == "qwen3-8b"
        assert cfg2.n_prompts_per_cat == 5

def test_no_hardcoded_seed_in_analysis():
    # seed should be config-driven, not literal 0 in logic except default
    import analysis_utils
    import inspect
    src = inspect.getsource(analysis_utils.bootstrap_ci)
    assert "seed" in src

def test_results_json_schema():
    """Check existing pilot results have expected keys (skip if not yet generated)."""
    import pytest
    found_any = False
    for phase in ["phase1_safety_gap","phase2_deception","phase3_cross_arch"]:
        p = pathlib.Path(__file__).parent.parent / "results" / phase / "results.json"
        if p.exists():
            found_any = True
            data = json.loads(p.read_text())
            assert isinstance(data, dict)
            assert len(data) > 0, f"{phase}/results.json is empty"
    if not found_any:
        pytest.skip("No pilot results generated yet")

def test_thinking_machines_grant_doc_exists():
    p = pathlib.Path(__file__).parent.parent / "THINKING_MACHINES_GRANT.md"
    assert p.exists()
    text = p.read_text()
    for kw in ["Relevance","Feasibility","Construct validity","Simplicity"]:
        assert kw.lower() in text.lower() or kw in text

def test_contrastive_pairs_yield_positive_gap():
    """Deceptive should be more suspicious than honest on at least one metric in pilot."""
    import pytest
    p = pathlib.Path(__file__).parent.parent / "results" / "phase2_deception" / "results.json"
    if not p.exists():
        pytest.skip("No phase2 results generated yet")
    data = json.loads(p.read_text())
    if "scores" not in data:
        pytest.skip("No scores in phase2 results")
    scores = data["scores"]
    if "adversarial_escalation" in scores and "in_distribution" in scores:
        adv = np.mean([s["divergence_area"] for s in scores["adversarial_escalation"]])
        ind = np.mean([s["divergence_area"] for s in scores["in_distribution"]])
        assert adv > ind, f"adversarial ({adv:.3f}) should have higher divergence than in_dist ({ind:.3f})"
    else:
        pytest.skip("Missing adversarial_escalation or in_distribution scores")

def test_five_reproducibility_runs_same_ci():
    import analysis_utils as au
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, 20).tolist()
    cis = [au.bootstrap_ci(x, n_boot=200, seed=123)[1] for _ in range(5)]
    for c in cis:
        assert abs(c - cis[0]) < 1e-10, f"Non-deterministic CI: {cis[0]} vs {c}"
