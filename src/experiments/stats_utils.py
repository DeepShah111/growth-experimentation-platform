"""Small, tested statistics helpers used by the studies. No hidden magic: every formula is here."""
from __future__ import annotations

import numpy as np
from scipy import stats

Z = stats.norm.ppf


# --------------------------------------------------------------------------- basic tests
def welch_t(y_t: np.ndarray, y_c: np.ndarray) -> tuple[float, float]:
    """Welch t statistic and two-sided p-value (fast, numpy only)."""
    nt, nc = len(y_t), len(y_c)
    vt, vc = y_t.var(ddof=1), y_c.var(ddof=1)
    se = np.sqrt(vt / nt + vc / nc)
    t = (y_t.mean() - y_c.mean()) / se
    return float(t), float(2 * stats.norm.sf(abs(t)))   # large n: normal approx of t


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y - theta*(x - mean(x)),  theta = cov(y,x)/var(x). Keeps the mean, shrinks the variance."""
    vx = np.var(x, ddof=1)
    if vx == 0:
        return y.astype(float).copy()
    theta = np.cov(y, x, ddof=1)[0, 1] / vx
    return y - theta * (x - x.mean())


# ------------------------------------------------------------------------- power maths
def required_n_per_arm(rel_lift: float, mean: float, sd: float, alpha: float = 0.05,
                       power: float = 0.8, rho: float = 0.0) -> float:
    """Users needed PER ARM to detect a relative lift. rho = corr(outcome, CUPED covariate)."""
    sd_eff = sd * np.sqrt(1 - rho**2)
    delta = rel_lift * mean
    return 2 * ((Z(1 - alpha / 2) + Z(power)) * sd_eff / delta) ** 2


def mde_rel(n_per_arm: float, mean: float, sd: float, alpha: float = 0.05,
            power: float = 0.8, rho: float = 0.0) -> float:
    """Smallest relative lift detectable with n users per arm."""
    sd_eff = sd * np.sqrt(1 - rho**2)
    return (Z(1 - alpha / 2) + Z(power)) * np.sqrt(2) * sd_eff / (np.sqrt(n_per_arm) * mean)


def analytic_power(rel_lift: float, n_per_arm: float, mean: float, sd: float,
                   alpha: float = 0.05, rho: float = 0.0) -> float:
    sd_eff = sd * np.sqrt(1 - rho**2)
    se = np.sqrt(2) * sd_eff / np.sqrt(n_per_arm)
    d = rel_lift * mean / se
    return float(stats.norm.sf(Z(1 - alpha / 2) - d) + stats.norm.cdf(-Z(1 - alpha / 2) - d))


# ----------------------------------------------------------- sequential (peeking) bounds
def obrien_fleming_bounds(n_looks: int, alpha: float = 0.05, n_sims: int = 200_000,
                          seed: int = 0) -> np.ndarray:
    """O'Brien-Fleming-style z boundaries  c*sqrt(K/k) at equally spaced looks.

    c is calibrated by Monte Carlo so that the overall false-positive rate across all looks
    equals alpha (common random numbers + bisection). Strict early, relaxed at the end.
    """
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((n_sims, n_looks))
    k = np.arange(1, n_looks + 1)
    zpath = np.abs(np.cumsum(inc, axis=1)) / np.sqrt(k)        # |Z_k| of partial sums
    shape = np.sqrt(n_looks / k)

    def fpr(c):
        return float((zpath > c * shape).any(axis=1).mean())

    lo, hi = 1.0, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fpr(mid) > alpha else (lo, mid)
    return ((lo + hi) / 2) * shape


# ------------------------------------------------------------------- multiple testing
def holm(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down: returns boolean 'reject' array (controls family-wise error)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    reject = np.zeros(m, bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values: reject at level alpha exactly when the adjusted value is <= alpha."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (m - rank)))
        adj[idx] = running
    return adj


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH step-up: returns boolean 'reject' array (controls false discovery rate)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= alpha * (np.arange(1, m + 1) / m)
    reject = np.zeros(m, bool)
    if passed.any():
        cutoff = np.max(np.where(passed)[0])
        reject[order[: cutoff + 1]] = True
    return reject


def bonferroni(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p, float)
    return p <= alpha / len(p)


# ------------------------------------------------------------------- lift, adjustment, money
def lift_delta(y_t: np.ndarray, y_c: np.ndarray) -> tuple[float, float]:
    """Relative lift of treatment over control and the 95% half-width of its interval (delta method)."""
    mt, mc = y_t.mean(), y_c.mean()
    se_t, se_c = y_t.std(ddof=1) / np.sqrt(len(y_t)), y_c.std(ddof=1) / np.sqrt(len(y_c))
    with np.errstate(divide="ignore", invalid="ignore"):                 # a zero control mean gives inf/nan, not a crash
        return float(mt / mc - 1), float(1.96 * np.sqrt((se_t / mc) ** 2 + (mt * se_c / mc**2) ** 2))


def regression_adjust(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Multi-covariate CUPED: y - (X - mean(X)) @ theta, theta = OLS of y on X (arms pooled).

    The adjustment has mean zero in expectation, so the lift stays unbiased while the variance shrinks.
    Returns (adjusted outcome, R^2 of the covariates). With one covariate this equals `cuped_adjust`.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    Xc = X - X.mean(0)
    keep = Xc.std(0) > 0                                    # constant columns carry no information
    if not keep.any() or y.std() == 0:
        return y.copy(), 0.0
    Xc = Xc[:, keep]
    theta, *_ = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)
    fitted = Xc @ theta
    r2 = float(np.var(fitted) / np.var(y))
    return y - fitted, min(max(r2, 0.0), 0.999999)


def net_value_per_user(y_treat: np.ndarray, y_ctrl: np.ndarray, value_per_order: float, cost_per_treated: float):
    """Profit per user of rolling the change out to everyone: value x (extra orders per user) - cost. Welch 95% CI."""
    d = y_treat.mean() - y_ctrl.mean()
    se = np.sqrt(y_treat.var(ddof=1) / len(y_treat) + y_ctrl.var(ddof=1) / len(y_ctrl))
    return value_per_order * d - cost_per_treated, value_per_order * (d - 1.96 * se) - cost_per_treated, \
        value_per_order * (d + 1.96 * se) - cost_per_treated
