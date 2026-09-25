"""Tests that no hardcoded magic numbers remain outside config."""
import pathlib, re

ROOT = pathlib.Path(__file__).parent.parent
ALLOW = {"config.py", "analysis_utils.py"}  # allowed to define defaults

def _files():
    return [p for p in ROOT.rglob("*.py") if p.name not in ALLOW and "tests" not in str(p) and ".git" not in str(p) and "jacobian-lens" not in str(p)]

def test_no_hardcoded_heuristic_thresholds():
    # deception_detector should not contain literal 3.0, 0.3 etc outside config import
    for p in _files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        # look for hardcoded classify thresholds not via config
        if "classify_heuristic" in text:
            assert "thresholds.divergence_area" in text or "thresholds.mid_late_gap" in text, f"{p.name} still hardcodes thresholds"
            # ensure no literal  > 3.0 in that file except in config
            assert text.count("> 3.0") == 0, f"{p.name} has hardcoded 3.0"

def test_no_hardcoded_artifact_thresholds():
    for p in _files():
        if "cross_arch" in p.name or "safety_gap" in p.name:
            text = p.read_text(encoding="utf-8", errors="ignore")
            # should use config thresholds, not literal 0.1 and 0.3
            if "artifact_exists" in text:
                assert "_thr.artifact_peak_min" in text or "thresholds.artifact" in text, f"{p.name} hardcodes artifact thresholds"

def test_no_hardcoded_layers():
    for p in _files():
        if p.name in ("safety_gap.py", "deception_detector.py", "cross_arch.py"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            # should derive layers dynamically (not hardcoded list(range(28)) with literal 28)
            has_dynamic = ("analysis_layers" in text) or ("range(n_layers)" in text) or ("range( n_layers" in text)
            assert has_dynamic, f"{p.name} missing dynamic layers"
            assert "range(28)" not in text or "range(n_layers)" in text, f"{p.name} still hardcodes 28"

def test_no_hardcoded_paths():
    for p in _files():
        if p.name in ("safety_gap.py", "deception_detector.py", "cross_arch.py"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            # should import from config, not hardcode C:\...
            assert "DEFAULT_MODEL_DIR" in text or "get_config" in text, f"{p.name} hardcodes paths"

def test_config_has_all_thresholds():
    import config
    thr = config.ThresholdConfig()
    for field in ["divergence_area","mid_late_gap","trajectory_var","final_surprise","suspicious_frac","uncertain_frac","artifact_peak_min","artifact_late_ratio","patch_alphas","high_divergence_cutoff","artifact_advantage_floor"]:
        assert hasattr(thr, field)

def test_n_boot_not_hardcoded_in_phases():
    for name in ["safety_gap.py","deception_detector.py","cross_arch.py"]:
        p = ROOT / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            # should reference _cfg.n_boot or cfg.n_boot or N_BOOT from config, not literal 5000
            # count literal 5000 usages outside of default param values
            lines = text.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if "5000" in stripped and not stripped.startswith("#") and "default" not in stripped and "n_boot=" not in stripped:
                    assert False, f"{name}:{i} hardcodes 5000: {stripped}"
    assert True
