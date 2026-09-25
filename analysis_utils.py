"""
analysis_utils: rigorous statistical tools for J-Lens safety evaluation.

Provides bootstrap CIs, permutation tests, effect sizes, ROC, and
publication-ready plotting helpers. All functions are deterministic
when seed is set.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Sequence

# numpy compat: np.trapezoid requires NumPy >= 2.0, np.trapz is deprecated
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ─── Core similarity ─────────────────────────────────────────────────────────

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.float().flatten(), b.float().flatten()
    n1, n2 = torch.norm(a_f), torch.norm(b_f)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return (torch.dot(a_f, b_f) / (n1 * n2)).item()


def free_mem():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ─── Bootstrap ───────────────────────────────────────────────────────────────

def bootstrap_ci(
    x: Sequence[float],
    stat_fn=np.mean,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for any statistic. Returns (estimate, lo, hi)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    est = float(stat_fn(x))
    if len(x) == 1:
        return est, est, est
    boots = np.array([stat_fn(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1-ci)/2*100, (1+ci)/2*100])
    return est, float(lo), float(hi)


def bootstrap_diff_ci(
    a: Sequence[float], b: Sequence[float],
    n_boot: int = 5000, ci: float = 0.95, seed: int = 0,
) -> tuple[float, float, float, float]:
    """Bootstrap CI for mean(a) - mean(b). Returns (diff, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a)==0 or len(b)==0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diff = float(np.mean(a) - np.mean(b))
    if len(a) == 1 and len(b) == 1:
        return diff, diff, diff, 1.0
    boots = np.array([
        float(np.mean(rng.choice(a, len(a), True)) - np.mean(rng.choice(b, len(b), True)))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [(1-ci)/2*100, (1+ci)/2*100])
    # two-sided p via bootstrap permutation-style test
    p_le = (boots <= 0).mean()
    p = float(2 * min(p_le, 1.0 - p_le)) if diff != 0 else 1.0
    return diff, float(lo), float(hi), float(p)

# ─── Permutation test ────────────────────────────────────────────────────────

def permutation_test(
    a: Sequence[float], b: Sequence[float],
    n_perm: int = 10000, seed: int = 0,
) -> tuple[float, float]:
    """Two-sided permutation test for difference in means. Returns (diff, p)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a)==0 or len(b)==0:
        return float("nan"), float("nan")
    obs = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_diff = float(np.mean(pooled[:len(a)]) - np.mean(pooled[len(a):]))
        if abs(perm_diff) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)  # smoothed
    return obs, float(p)

# ─── Effect sizes ────────────────────────────────────────────────────────────

def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for two independent samples."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    # pooled std
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta (non-parametric effect size, robust to non-normal).
    O(n log n) via sort — avoids O(n²) memory from broadcasting."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a)==0 or len(b)==0:
        return float("nan")
    b_sorted = np.sort(b)
    gt = int(np.sum(np.searchsorted(b_sorted, a, side='left')))
    lt = int(np.sum(len(b) - np.searchsorted(b_sorted, a, side='right')))
    return float((gt - lt) / (len(a)*len(b)))

# ─── Multiple testing ────────────────────────────────────────────────────────

def bonferroni(pvals: Sequence[float]) -> list[float]:
    n = len(pvals)
    return [min(1.0, float(p)*n) for p in pvals]

def benjamini_hochberg(pvals: Sequence[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, float)
    ranked[order] = np.arange(1, n+1)
    # BH: p * n / rank, then cumulative min
    adj = np.array(pvals, float) * n / ranked
    # ensure monotonic
    adj_sorted = adj[order]
    for i in range(n-2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i+1])
    result = np.empty(n, float)
    result[order] = np.minimum(1.0, adj_sorted)
    return result.tolist()

# ─── ROC / Probe ─────────────────────────────────────────────────────────────

def roc_auc_and_curve(y_true: Sequence[int], y_score: Sequence[float]):
    """ROC-AUC with correct tied-score handling (threshold grouping)."""
    y_true = np.asarray(y_true, int); y_score = np.asarray(y_score, float)
    if len(y_true) != len(y_score):
        raise ValueError(f"y_true ({len(y_true)}) and y_score ({len(y_score)}) must have same length")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score contains NaN/inf — ROC tie-grouping would hang on NaN (NaN != NaN)")
    P = int((y_true == 1).sum()); N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        return float("nan"), np.array([0, 1]), np.array([0, 1])
    if _trapz is None:
        raise ImportError("numpy needs trapezoid/trapz for ROC integration")
    # sort by score descending, stable
    order = np.argsort(-y_score, kind="mergesort")
    y_true_s = y_true[order]
    y_score_s = y_score[order]
    # group by unique score thresholds (including +inf and -inf)
    # standard ROC: thresholds are y_score values; tied scores count as one step
    tpr_list, fpr_list = [0.0], [0.0]
    tp, fp = 0, 0
    i = 0
    while i < len(y_score_s):
        # count all samples with same score
        score = y_score_s[i]
        # process block of ties together
        j = i
        while j < len(y_score_s) and y_score_s[j] == score:
            if y_true_s[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
        i = j
    # ensure (1,1) endpoint already included (last block does)
    tpr = np.array(tpr_list); fpr = np.array(fpr_list)
    # trapezoidal AUC: integral of tpr over fpr (fpr is x, tpr is y)
    # fpr is monotone by construction (grouped ties)
    auc = float(_trapz(tpr, fpr))
    return auc, fpr, tpr


def fit_linear_probe(
    X: np.ndarray, y: np.ndarray,
    cv: int = 5, seed: int = 0,
) -> dict:
    """L2 logistic probe with cross-validated ROC-AUC. Requires sklearn if available,
    falls back to centroid classifier."""
    X = np.asarray(X, float); y = np.asarray(y, int)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"X must be 2-D with len(X)==len(y), got {X.shape} and {len(y)}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN/inf")
    if len(np.unique(y)) != 2:
        raise ValueError(f"probe needs exactly 2 classes, got {np.unique(y).tolist()}")
    min_class = int(min((y == c).sum() for c in np.unique(y)))
    cv = max(2, min(cv, len(y), min_class))
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(X, y):
            # handle case where train OR test has single class
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                aucs.append(0.5)
                continue
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:,1]
            aucs.append(float(roc_auc_score(y[te], prob)))
        # fit on full data for weights
        clf_full = LogisticRegression(max_iter=1000).fit(X, y) if len(np.unique(y))>1 else None
        return {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
                "aucs": aucs, "coef": clf_full.coef_[0] if clf_full is not None else None}
    except ImportError:
        # centroid fallback: distance to class means
        if len(np.unique(y)) < 2:
            return {"cv_auc_mean": 0.5, "cv_auc_std": 0.0, "aucs": [0.5]*cv, "coef": None}
        m0 = X[y==0].mean(0); m1 = X[y==1].mean(0)
        # score = projection onto (m1 - m0)
        direction = m1 - m0
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return {"cv_auc_mean": 0.5, "cv_auc_std": 0.0, "aucs": [0.5]*cv, "coef": direction}
        scores = X @ direction
        auc, _, _ = roc_auc_and_curve(y, scores)
        return {"cv_auc_mean": auc, "cv_auc_std": 0.0, "aucs": [auc]*cv, "coef": direction}

# ─── CKA / subspace ──────────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation matrices (n × d)."""
    # center
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # Gram
    dot = float(np.linalg.norm(Y.T @ X, ord='fro')**2)
    norm_x = float(np.linalg.norm(X.T @ X, ord='fro'))
    norm_y = float(np.linalg.norm(Y.T @ Y, ord='fro'))
    if norm_x < 1e-12 or norm_y < 1e-12:
        return 0.0
    return dot / (norm_x * norm_y)


def subspace_overlap(Ja: np.ndarray, Jb: np.ndarray, k: int = 32) -> float:
    """Top-k subspace overlap via SVD: mean squared cosine of singular vectors."""
    # Ja, Jb are Jacobians d×d — must share the row dimension for Ua_k.T @ Ub_k
    Ja = np.asarray(Ja, float); Jb = np.asarray(Jb, float)
    if Ja.ndim != 2 or Jb.ndim != 2:
        raise ValueError(f"Jacobians must be 2-D, got {Ja.shape} and {Jb.shape}")
    if Ja.shape[0] != Jb.shape[0]:
        raise ValueError(f"Jacobian row dims must match for subspace overlap, got {Ja.shape} vs {Jb.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    Ua, _, _ = np.linalg.svd(Ja, full_matrices=False)
    Ub, _, _ = np.linalg.svd(Jb, full_matrices=False)
    # overlap of top-k (clamped to matrix rank)
    k = min(k, Ua.shape[1], Ub.shape[1])
    Ua_k = Ua[:, :k]; Ub_k = Ub[:, :k]
    # squared cosines of principal angles: singular values of Ua_k^T Ub_k
    M = Ua_k.T @ Ub_k
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(s**2))
