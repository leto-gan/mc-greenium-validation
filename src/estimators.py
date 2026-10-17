"""Estimators for the Monte Carlo design-validation study.

All estimators are implemented in pure NumPy (no statsmodels/linearmodels
dependency):

- two-way (bond and day) fixed effects by within transformation
  (exact closed form for balanced panels; iterative weighted demeaning for
  unbalanced panels and/or weighted estimators);
- issuer-clustered standard errors via a hand-built sandwich with
  cluster-summed scores (clusters = the 150 issuers);
- entropy balancing weights (scipy SLSQP) and coarsened exact matching
  (CEM) weights for the matched-sample variants.

Estimator list (framework Section 4):
    E1  TWFE triple-difference, binary regime R1 = EU x post-Level-2
    EB  E1 on entropy-balanced sample
    CEM E1 on CEM-matched sample
    E6  TWFE with continuous pre-determined treatment intensity R2
        (Article 8/9 ownership share), preferred specification
The dynamic event-study diagnostic (framework Figure M3) is model-free:
issuer-pair spread differences averaged in 5-day bins around shocks,
contrasted between treated and comparison groups.
"""

import numpy as np
from scipy.optimize import minimize

POST_START = 400   # duplicated from dgp to keep this module importable alone


# ---------------------------------------------------------------------------
# Within transformation
# ---------------------------------------------------------------------------

def demean2d(M, row_w=None, mask=None, n_iter=60, tol=1e-9):
    """Two-way (row + column) demeaning of a (bond, day) matrix.

    Exact in one pass for the balanced unweighted case; iterative otherwise.
    `row_w` are bond-level observation weights; `mask` marks valid cells.
    """
    M = np.asarray(M, dtype=float)
    if row_w is None and mask is None:
        return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    W = np.ones(M.shape) if mask is None else mask.astype(float)
    if row_w is not None:
        W = W * row_w[:, None]
    out = M.copy()
    for _ in range(n_iter):
        rden = W.sum(1)
        r = (out * W).sum(1) / np.maximum(rden, 1e-12)
        out = out - np.where(rden > 0, r, 0.0)[:, None]
        cden = W.sum(0)
        c = (out * W).sum(0) / np.maximum(cden, 1e-12)
        out = out - np.where(cden > 0, c, 0.0)[None, :]
        if np.abs(r).max(initial=0.0) < tol and np.abs(c).max(initial=0.0) < tol:
            break
    return out


# ---------------------------------------------------------------------------
# TWFE with issuer-clustered sandwich standard errors
# ---------------------------------------------------------------------------

def twfe_fit(y, X_list, row_w=None, mask=None):
    """Fit y = X'beta + bond FE + day FE by within-transformed (W)LS.

    Returns (beta, se) with issuer-clustered standard errors.
    Issuers are contiguous bond pairs (bonds 2j and 2j+1 of issuer j).
    """
    B, T = y.shape
    assert B % 2 == 0
    G = B // 2
    mats = [y, *X_list]
    dm = [demean2d(m, row_w=row_w, mask=mask) for m in mats]
    sel = np.ones((B, T), bool) if mask is None else mask
    Y = dm[0][sel]
    Xs = np.column_stack([d[sel] for d in dm[1:]])
    n, p = Xs.shape
    if row_w is None:
        wsel = np.ones(n)
    else:
        wgrid = np.broadcast_to(row_w.astype(float)[:, None], (B, T))
        wsel = wgrid[sel]
    sq = np.sqrt(wsel)
    beta, *_ = np.linalg.lstsq(Xs * sq[:, None], Y * sq, rcond=None)
    resid = Y - Xs @ beta

    # cluster-summed scores (masked cells contribute zero)
    sc = np.zeros((B, T, p))
    sc[sel] = Xs * (wsel * resid)[:, None]
    S = sc.reshape(G, -1, p).sum(1)
    active = int((np.abs(S).sum(1) > 0).sum())   # clusters with observations
    meat = S.T @ S
    XtWX = (Xs * wsel[:, None]).T @ Xs
    bread = np.linalg.inv(XtWX)
    corr = (active / (active - 1.0)) * ((n - 1.0) / (n - p))
    V = bread @ meat @ bread * corr
    return beta, np.sqrt(np.diag(V))


# ---------------------------------------------------------------------------
# Matching weights (computed once per fixed design)
# ---------------------------------------------------------------------------

def entropy_balance_weights(design):
    """Entropy-balance conventional bonds to green-bond covariate moments."""
    cov = np.column_stack([
        design.rating.astype(float),
        design.maturity,
        design.logsize,
        design.maturity ** 2,
        design.logsize ** 2,
    ])
    t = design.green
    target = cov[t].mean(0)
    Xc = cov[~t]
    n_c = Xc.shape[0]
    q0 = np.full(n_c, 1.0 / n_c)

    def obj(w):
        return (w * np.log(np.maximum(w, 1e-12) / q0)).sum()

    def grad(w):
        return np.log(np.maximum(w, 1e-12) / q0) + 1.0

    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0,
         "jac": lambda w: np.ones_like(w)},
        {"type": "eq", "fun": lambda w: Xc.T @ w - target,
         "jac": lambda w: Xc.T},
    ]
    res = minimize(obj, q0, jac=grad, constraints=cons,
                   bounds=[(1e-9, None)] * n_c, method="SLSQP",
                   options={"maxiter": 1000, "ftol": 1e-12})
    # SLSQP sometimes reports an internal iteration limit even when the
    # balance constraints are satisfied; judge convergence numerically.
    balanced = (np.abs(res.x.sum() - 1.0) < 1e-9 and
                np.abs(Xc.T @ res.x - target).max() < 1e-6)
    w = np.ones(design.n_bonds)
    w[~t] = res.x * n_c          # normalise control weights to mean 1
    return w, bool(res.success or balanced)


def cem_weights(design):
    """Coarsened exact matching: rating x maturity tercile x size tercile.

    Returns (weights, keep): bonds in strata without both green and
    conventional bonds are dropped.
    """
    B = design.n_bonds
    mq = np.quantile(design.maturity, [1 / 3, 2 / 3])
    sq = np.quantile(design.logsize, [1 / 3, 2 / 3])
    mat_t = np.digitize(design.maturity, mq)
    siz_t = np.digitize(design.logsize, sq)
    stratum = design.rating * 9 + mat_t * 3 + siz_t
    w = np.zeros(B)
    keep = np.zeros(B, dtype=bool)
    for s in np.unique(stratum):
        idx = stratum == s
        ng = int((idx & design.green).sum())
        nc = int((idx & ~design.green).sum())
        if ng > 0 and nc > 0:
            keep |= idx
            w[idx & design.green] = 1.0
            w[idx & ~design.green] = ng / nc
    cm = w[~design.green & keep].mean()
    w[~design.green & keep] /= cm
    return w, keep


# ---------------------------------------------------------------------------
# Estimator runner for one replication
# ---------------------------------------------------------------------------

def run_estimators(rep, design, cfg):
    """Run the estimator set appropriate for the scenario's regime type.

    Returns dict: estimator -> (phi_hat, phi_se, tau_hat, tau_se).
    """
    from dgp import T_DAYS
    y, shock, mask = rep["y"], rep["shock"], rep["mask"]
    post = np.arange(T_DAYS) > POST_START
    Gm = design.green[:, None].astype(float)
    S = shock.astype(float)
    out = {}
    if cfg["regime"] == "binary":
        R = (design.bond_eu[:, None] & post[None, :]).astype(float)
        X = [Gm * S * R, Gm * S, Gm * R, S * R, S]
        b, se = twfe_fit(y, X, mask=mask)
        out["E1"] = (b[0], se[0], b[1], se[1])
        b, se = twfe_fit(y, X, row_w=design.w_eb, mask=mask)
        out["EB"] = (b[0], se[0], b[1], se[1])
        m_cem = mask & design.keep_cem[:, None]
        b, se = twfe_fit(y, X, row_w=design.w_cem, mask=m_cem)
        out["CEM"] = (b[0], se[0], b[1], se[1])
    else:
        # w is defined (non-zero) only for green bonds, so S x w is collinear
        # with G x S x w and G x w is absorbed by bond FE; the identified
        # intensity specification is y = b1*G*S*w + b2*G*S + kappa*S + FE.
        wm = design.w[:, None]
        X = [Gm * S * wm, Gm * S, S]
        b, se = twfe_fit(y, X, mask=mask)
        out["E6"] = (b[0], se[0], b[1], se[1])
    return out


# ---------------------------------------------------------------------------
# Model-free dynamic event-study diagnostic (framework Figure M3)
# ---------------------------------------------------------------------------

def event_study_ddd(rep, design, regime, bin_edges):
    """Dynamic DDD coefficients in 5-day bins around each shock.

    Pair spread s_jt = y_green - y_conventional within issuer j.
    For each post-Level-2 event, the treated group is compared with the
    comparison group among exposed issuers of the hit region:
      binary regime:    treated = EU issuers, comparison = non-EU issuers
      intensity regime: treated = above-median ownership w, rest comparison
    Returns array of shape (n_bins,) with NaN where a cell is empty.
    """
    y = rep["y"]
    s = y[0::2] - y[1::2]                    # (issuers, days)
    n_bins = len(bin_edges) - 1
    coefs = np.full(n_bins, np.nan)
    t_idx = np.arange(s.shape[1])
    treat_issuers = (design.eu if regime == "binary"
                     else design.w[0::2] > design.w_median)
    rows_t, rows_c = {b: [] for b in range(n_bins)}, {b: [] for b in range(n_bins)}
    for d, r in rep["events"]:
        if d <= POST_START:
            continue                          # no regime variation pre-Level-2
        exposed = design.region == r
        tr = exposed & treat_issuers
        co = exposed & ~treat_issuers
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            days = d + np.arange(lo, hi)
            if days.min() < 0 or days.max() >= s.shape[1]:
                continue
            if tr.any():
                rows_t[b].append(s[np.ix_(tr, days)].mean())
            if co.any():
                rows_c[b].append(s[np.ix_(co, days)].mean())
    for b in range(n_bins):
        if rows_t[b] and rows_c[b]:
            coefs[b] = np.mean(rows_t[b]) - np.mean(rows_c[b])
    return coefs
