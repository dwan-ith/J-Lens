"""Tests for analysis_utils — bootstrap, permutation, effect sizes, ROC, CKA."""
import numpy as np
import torch
import analysis_utils as au

def test_cosine_sim_identity():
    a = torch.randn(32)
    assert abs(au.cosine_sim(a, a) - 1.0) < 1e-5

def test_cosine_sim_orthogonal():
    a = torch.tensor([1.,0.]); b = torch.tensor([0.,1.])
    assert abs(au.cosine_sim(a,b)) < 1e-6

def test_cosine_sim_zero_vector():
    a = torch.zeros(10); b = torch.randn(10)
    assert abs(au.cosine_sim(a,b)) < 1e-8

def test_cosine_sim_negative():
    a = torch.tensor([1., 0.]); b = torch.tensor([-1., 0.])
    assert abs(au.cosine_sim(a, b) - (-1.0)) < 1e-6

def test_bootstrap_ci_basic():
    x = [0.5]*10
    est, lo, hi = au.bootstrap_ci(x, n_boot=200, seed=0)
    assert lo <= est <= hi
    assert abs(est-0.5) < 1e-6

def test_bootstrap_ci_empty():
    est, lo, hi = au.bootstrap_ci([], n_boot=100)
    assert np.isnan(est) and np.isnan(lo) and np.isnan(hi)

def test_bootstrap_ci_single():
    est, lo, hi = au.bootstrap_ci([1.0], n_boot=100)
    assert abs(est - 1.0) < 1e-10 and abs(lo - 1.0) < 1e-10 and abs(hi - 1.0) < 1e-10

def test_bootstrap_diff_ci():
    a = np.random.default_rng(0).normal(1.0, 0.1, 20)
    b = np.random.default_rng(1).normal(0.0, 0.1, 20)
    diff, lo, hi, p = au.bootstrap_diff_ci(a,b, n_boot=500, seed=0)
    assert diff > 0.5
    assert lo > 0  # should exclude 0
    assert p < 0.05

def test_permutation_test_significant():
    rng = np.random.default_rng(0)
    a = rng.normal(5, 1, 30); b = rng.normal(0, 1, 30)
    diff, p = au.permutation_test(a,b, n_perm=1000, seed=0)
    assert p < 0.01

def test_permutation_test_null():
    rng = np.random.default_rng(0)
    a = rng.normal(0,1,30); b = rng.normal(0,1,30)
    diff, p = au.permutation_test(a,b, n_perm=1000, seed=0)
    assert p > 0.05

def test_cohens_d():
    a = [1.0,1.1,0.9,1.2]; b = [0.0,0.1,-0.1,0.2]
    d = au.cohens_d(a,b)
    assert d > 2.0  # large effect

def test_cohens_d_equal():
    a = [1,2,3]; b = [1,2,3]
    assert abs(au.cohens_d(a,b)) < 1e-6

def test_cliffs_delta():
    a = [3,4,5]; b = [1,2,3]
    delta = au.cliffs_delta(a,b)
    # gt=8 (3>1,3>2, 4>1,4>2,4>3, 5>1,5>2,5>3), lt=0 → delta=8/9
    assert abs(delta - 8/9) < 1e-6

def test_cliffs_delta_negative():
    a = [1,2,3]; b = [3,4,5]
    delta = au.cliffs_delta(a,b)
    assert delta < 0

def test_bonferroni():
    pvals = [0.01, 0.02, 0.03]
    adj = au.bonferroni(pvals)
    assert adj[0] == 0.03
    assert all(v <= 1.0 for v in adj)

def test_benjamini_hochberg():
    pvals = [0.01, 0.02, 0.5]
    adj = au.benjamini_hochberg(pvals)
    assert len(adj)==3
    assert adj[0] < adj[2]
    assert all(0 <= v <= 1 for v in adj)
    # correct BH: [0.03, 0.03, 0.5]
    assert abs(adj[0] - 0.03) < 1e-10
    assert abs(adj[1] - 0.03) < 1e-10
    assert abs(adj[2] - 0.5) < 1e-10

def test_roc_auc_perfect():
    y_true = [0,0,1,1]; y_score = [0.1,0.2,0.8,0.9]
    auc, fpr, tpr = au.roc_auc_and_curve(y_true, y_score)
    assert abs(auc - 1.0) < 1e-6

def test_roc_auc_random():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0,2,100); y_score = rng.random(100)
    auc, _, _ = au.roc_auc_and_curve(y_true, y_score)
    assert 0.3 < auc < 0.7

def test_fit_linear_probe():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-1,0.5,(20,4)), rng.normal(1,0.5,(20,4))])
    y = np.array([0]*20+[1]*20)
    res = au.fit_linear_probe(X,y, cv=3, seed=0)
    assert res["cv_auc_mean"] > 0.8

def test_linear_cka_identical():
    rng = np.random.default_rng(0)
    X = rng.random((20,8))
    assert abs(au.linear_cka(X, X) - 1.0) < 1e-6

def test_linear_cka_orthogonal():
    X = np.eye(10); Y = np.random.default_rng(0).random((10,10))
    cka = au.linear_cka(X, Y)
    assert 0 <= cka <= 1

def test_subspace_overlap_identical():
    rng = np.random.default_rng(0)
    J = rng.random((16,16))
    assert abs(au.subspace_overlap(J, J, k=4) - 1.0) < 1e-5

def test_subspace_overlap_random():
    rng = np.random.default_rng(0)
    Ja = rng.random((16,16)); Jb = rng.random((16,16))
    ov = au.subspace_overlap(Ja, Jb, k=4)
    assert 0 <= ov <= 1

def test_subspace_overlap_k_clamped():
    rng = np.random.default_rng(0)
    Ja = rng.random((8,8)); Jb = rng.random((8,8))
    # k > matrix dimension should be clamped, not crash
    ov = au.subspace_overlap(Ja, Jb, k=32)
    assert 0 <= ov <= 1

def test_bootstrap_diff_ci_single_element():
    diff, lo, hi, p = au.bootstrap_diff_ci([1.0], [0.5], n_boot=100)
    assert abs(diff - 0.5) < 1e-10
    assert p == 1.0

def test_cohens_d_single_element():
    assert np.isnan(au.cohens_d([1.0], [2.0]))

def test_cohens_d_zero_variance_one_group():
    # one group has zero variance — pooled std from other group, d well-defined
    d = au.cohens_d([5,5,5], [1,2,3])
    assert abs(d) > 3.0

def test_cliffs_delta_empty():
    assert np.isnan(au.cliffs_delta([], [1,2,3]))

def test_roc_auc_all_positive():
    auc, fpr, tpr = au.roc_auc_and_curve([1,1,1], [0.5,0.6,0.7])
    assert np.isnan(auc)
