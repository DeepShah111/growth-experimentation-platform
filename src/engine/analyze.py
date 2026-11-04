"""Analyse ANY randomised experiment table: one row per user, an arm label, an outcome, optional pre-experiment covariates.

    result = analyze_experiment(df, variant="segment", outcome="spend", covariates=["history", "recency"],
                                cfg=AnalysisConfig(control="No E-Mail", min_effect=0.03))
    result["arms"]      # one row per treatment arm vs control, with the decision and the reasons
    result["srm"]       # sample-ratio check across ALL arms

What it does, in the same order as the project scorecard:
  1. TRUST     chi-square test of the arm sizes against the planned split (default: equal shares).
  2. EVIDENCE  relative lift vs control, naive and covariate-adjusted (multi-covariate CUPED), delta-method interval,
               Welch p-value, Holm correction across the treatment arms.
  3. POWER     minimum detectable lift from the actual sample and variance.
  4. SAFETY    optional guardrail metric: significantly negative blocks shipping.
  5. ECONOMICS optional: value per unit of outcome and cost per treated user -> profit per user with an interval.
The decision itself comes from `src.engine.decision.decide`, so the notebooks, the API and the app can never disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.engine.decision import MIN_EFFECT, SRM_ALPHA, decide
from src.experiments.stats_utils import holm, holm_adjust, lift_delta, mde_rel, net_value_per_user, regression_adjust


@dataclass(frozen=True)
class AnalysisConfig:
    control: str = "control"
    alpha: float = 0.05                              # family-wise error rate (Holm) across treatment arms
    srm_alpha: float = SRM_ALPHA
    min_effect: float = MIN_EFFECT                   # smallest relative lift worth shipping (business assumption)
    power: float = 0.8
    planned_split: dict | None = None                # arm -> planned share; None means equal shares
    value_per_unit: float | None = None              # money per unit of the outcome; with cost_per_treated switches economics on
    cost_per_treated: float = 0.0
    currency: str = "INR"
    guardrail: str | None = None                     # name of a second outcome column that must not get worse
    extra: dict = field(default_factory=dict)


class AnalysisError(ValueError):
    """The table cannot be analysed (missing column, no control rows, ...). The message is meant for the user."""


# ------------------------------------------------------------------------------------------ checks
def srm_test(counts: dict, planned_split: dict | None = None) -> dict:
    """Chi-square goodness of fit of the observed arm sizes against the planned shares."""
    arms = list(counts)
    obs = np.array([counts[a] for a in arms], float)
    if planned_split is None:
        share = np.full(len(arms), 1 / len(arms))
    else:
        missing = [a for a in arms if a not in planned_split]
        if missing:
            raise AnalysisError(f"planned_split has no share for arm(s): {missing}")
        share = np.array([planned_split[a] for a in arms], float)
        share = share / share.sum()
    p = float(stats.chisquare(obs, obs.sum() * share).pvalue) if len(arms) > 1 else 1.0
    return {"p_value": p, "counts": {a: int(c) for a, c in counts.items()},
            "expected": {a: float(e) for a, e in zip(arms, obs.sum() * share)}}


def _clean(df: pd.DataFrame, variant: str, outcome: str, covariates, guardrail) -> tuple[pd.DataFrame, list[str], list[str]]:
    notes: list[str] = []
    need = [variant, outcome, *covariates, *([guardrail] if guardrail else [])]
    absent = [c for c in need if c not in df.columns]
    if absent:
        raise AnalysisError(f"column(s) not in the table: {absent}")
    for c in (outcome, *covariates, *([guardrail] if guardrail else [])):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise AnalysisError(f"column '{c}' must be numeric (0/1 flags and counts are fine)")
    if outcome in covariates:
        raise AnalysisError("the outcome cannot also be a covariate")
    d = df[need].copy()
    d[variant] = d[variant].astype(str)
    n0 = len(d)
    d = d.dropna(subset=[variant, outcome])
    if len(d) < n0:
        notes.append(f"{n0 - len(d):,} rows dropped because the arm or the outcome was missing")
    for c in covariates:
        if d[c].isna().any():
            notes.append(f"covariate '{c}': {int(d[c].isna().sum()):,} missing values filled with the mean")
            d[c] = d[c].fillna(d[c].mean())
    if guardrail:
        d = d.dropna(subset=[guardrail])
    return d, list(covariates), notes


# --------------------------------------------------------------------------------- one comparison
def _compare(y_t: np.ndarray, y_c: np.ndarray, X_t: np.ndarray | None, X_c: np.ndarray | None) -> dict:
    """Naive and (if covariates) adjusted lift of one treatment arm over control."""
    lift, half = lift_delta(y_t, y_c)
    out = {"mean_treat": float(y_t.mean()), "mean_ctrl": float(y_c.mean()), "abs_effect": float(y_t.mean() - y_c.mean()),
           "naive_lift": lift, "naive_half": half, "naive_p": float(stats.ttest_ind(y_t, y_c, equal_var=False).pvalue),
           "r2": 0.0, "adj_lift": lift, "adj_half": half, "adj_p": float(stats.ttest_ind(y_t, y_c, equal_var=False).pvalue)}
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    out["abs_half"] = float(1.96 * se)
    if X_t is not None and X_c is not None and out["mean_ctrl"] > 0:
        y = np.r_[y_t, y_c]
        ya, r2 = regression_adjust(y, np.vstack([X_t, X_c]))
        yt_a, yc_a = ya[: len(y_t)], ya[len(y_t):]
        lift_a, half_a = lift_delta(yt_a, yc_a)
        out.update({"r2": r2, "adj_lift": lift_a, "adj_half": half_a,
                    "adj_p": float(stats.ttest_ind(yt_a, yc_a, equal_var=False).pvalue)})
        se_a = np.sqrt(yt_a.var(ddof=1) / len(yt_a) + yc_a.var(ddof=1) / len(yc_a))
        out["abs_half"] = float(1.96 * se_a)
        out["abs_effect"] = float(yt_a.mean() - yc_a.mean())
    return out


# ------------------------------------------------------------------------------------------- main
def analyze_experiment(df: pd.DataFrame, variant: str, outcome: str, covariates=(), cfg: AnalysisConfig | None = None) -> dict:
    cfg = cfg or AnalysisConfig()
    d, covs, notes = _clean(df, variant, outcome, list(covariates), cfg.guardrail)
    arms_all = sorted(d[variant].unique())
    if cfg.control not in arms_all:
        raise AnalysisError(f"control arm '{cfg.control}' not found; arms in the data: {arms_all}")
    treat_arms = [a for a in arms_all if a != cfg.control]
    if not treat_arms:
        raise AnalysisError("need at least one treatment arm besides the control")
    if (d[variant] == cfg.control).sum() < 30 or min((d[variant] == a).sum() for a in treat_arms) < 30:
        raise AnalysisError("every arm needs at least 30 users")

    srm = srm_test(d[variant].value_counts().to_dict(), cfg.planned_split)
    rows = []
    for arm in treat_arms:
        sub = d[d[variant].isin([cfg.control, arm])]
        is_t = (sub[variant] == arm).to_numpy()
        y = sub[outcome].to_numpy(float)
        X = sub[covs].to_numpy(float) if covs else None
        r = _compare(y[is_t], y[~is_t], X[is_t] if X is not None else None, X[~is_t] if X is not None else None)
        n_t, n_c = int(is_t.sum()), int((~is_t).sum())
        rho = float(np.sqrt(r["r2"]))
        usable = r["mean_ctrl"] > 0 and np.isfinite(r["adj_lift"])
        mde = float(mde_rel((n_t + n_c) / 2, r["mean_ctrl"], y[~is_t].std(ddof=1), power=cfg.power, rho=rho)) if usable else float("nan")
        row = {"arm": arm, "n_treat": n_t, "n_ctrl": n_c, **r, "lift": r["adj_lift"], "p_value": r["adj_p"],
               "ci_low": r["adj_lift"] - r["adj_half"], "ci_high": r["adj_lift"] + r["adj_half"],
               "ci_width_ratio": (r["adj_half"] / r["naive_half"]) if r["naive_half"] > 0 else 1.0,
               "mde": mde, "usable": bool(usable)}
        if cfg.guardrail:
            g = _compare(sub.loc[is_t, cfg.guardrail].to_numpy(float), sub.loc[~is_t, cfg.guardrail].to_numpy(float),
                         X[is_t] if X is not None else None, X[~is_t] if X is not None else None)
            row.update({"guardrail_lift": g["adj_lift"], "guardrail_breach": bool(g["adj_lift"] + g["adj_half"] < 0)})
        else:
            row.update({"guardrail_lift": 0.0, "guardrail_breach": False})
        nv = (float("nan"),) * 3
        if cfg.value_per_unit is not None:
            nv = net_value_per_user(y[is_t], y[~is_t], cfg.value_per_unit, cfg.cost_per_treated)
        row.update({"net_value": nv[0], "net_ci_low": nv[1], "net_ci_high": nv[2]})
        rows.append(row)

    t = pd.DataFrame(rows)
    ok = (t.usable & np.isfinite(t.p_value)).to_numpy()
    srm_ok = srm["p_value"] >= cfg.srm_alpha
    t["p_adj"] = np.nan
    t["significant"] = False
    if ok.any() and srm_ok:
        pv = t.loc[ok, "p_value"].to_numpy()
        t.loc[ok, "p_adj"] = holm_adjust(pv)
        t.loc[ok, "significant"] = holm(pv, cfg.alpha)
    t["underpowered"] = ~(t.mde <= cfg.min_effect)            # NaN MDE counts as underpowered
    decisions, reasons = [], []
    for rec in t.to_dict("records"):
        if not rec["usable"]:
            dec, why = "INCONCLUSIVE (underpowered)", ["control mean is zero or not finite, so a relative lift cannot be computed; "
                                                       "check the outcome column"]
        else:
            row = {**rec, "srm_p": srm["p_value"], "interaction_p": None, "overlap_with": "", "trend_p": None, "trend_slope": None,
                   "latest_week_lift": None, "latest_week_ci_half": None}
            dec, why = decide(row, min_effect=cfg.min_effect, srm_alpha=cfg.srm_alpha, currency=cfg.currency)
        decisions.append(dec)
        reasons.append(" | ".join(why))
    t["decision"], t["reasons"] = decisions, reasons
    if covs:
        notes.append(f"adjusted for {len(covs)} pre-experiment covariate(s); the interval is "
                     f"{1 - t.ci_width_ratio.mean():.0%} narrower on average than the naive one")
    if len(treat_arms) > 1:
        notes.append(f"Holm correction applied across {len(treat_arms)} treatment arms (family-wise alpha {cfg.alpha})")
    return {"srm": srm, "srm_ok": bool(srm_ok), "arms": t.drop(columns=["usable"]), "notes": notes,
            "control": cfg.control, "outcome": outcome, "covariates": covs, "n_rows": int(len(d))}


def to_jsonable(result: dict) -> dict:
    """Plain-Python version of a result (NaN -> None) for the API."""
    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    arms = [{k: clean(v) for k, v in rec.items()} for rec in result["arms"].to_dict("records")]
    return {"srm": {"p_value": clean(result["srm"]["p_value"]), "counts": result["srm"]["counts"],
                    "expected": {k: clean(v) for k, v in result["srm"]["expected"].items()}, "ok": result["srm_ok"]},
            "arms": arms, "notes": result["notes"], "control": result["control"], "outcome": result["outcome"],
            "covariates": result["covariates"], "n_rows": result["n_rows"]}
