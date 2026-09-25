"""Tests for safety_prompts — coverage and contrastive pairs."""
import safety_prompts as sp

def test_all_categories_count():
    cats = sp.get_all_categories()
    assert len(cats) == 8
    assert "in_distribution" in cats

def test_safety_categories_count():
    cats = sp.get_safety_categories()
    assert len(cats) == 7

def test_contrastive_pairs_min_edit():
    pairs = sp.get_contrastive_pairs()
    assert len(pairs) == 6
    for honest, deceptive in pairs:
        assert len(honest) >= 40
        assert len(deceptive) >= 40
        assert honest != deceptive
        # deceptive should differ semantically (at least 10 char diff or contains deception marker)
        assert abs(len(deceptive) - len(honest)) > 5 or any(w in deceptive.lower() for w in ["fabricated","false","distort","omit","recommendation","never happened","inaccurate"])

def test_contrastive_categories():
    cats = sp.get_contrastive_categories()
    assert "contrast_honest" in cats
    assert "contrast_deceptive" in cats
    assert len(cats["contrast_honest"]) == len(cats["contrast_deceptive"])

def test_prompts_length():
    for cat, prompts in sp.get_all_categories().items():
        for p in prompts:
            assert len(p) >= 40, f"{cat} prompt too short"

def test_no_empty_categories():
    for cat, prompts in sp.get_all_categories().items():
        assert len(prompts) > 0

def test_in_dist_distinct_from_safety():
    in_dist = set(sp.IN_DIST)
    safety = set(sp.SAFETY_DECEPTION)
    assert in_dist.isdisjoint(safety)

def test_adversarial_escalation_order():
    adv = sp.ADVERSARIAL_ESCALATION
    # should be 5 steps escalating
    assert len(adv) == 5
    assert "lock" in adv[0].lower()
