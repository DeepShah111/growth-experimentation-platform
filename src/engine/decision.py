"""The ship / iterate / hold decision rule. A pure function of one row of numbers, shared by every entry point.

Gates, in order: TRUST (sample ratio) -> EVIDENCE (Holm-adjusted significance) -> POWER (MDE vs the smallest lift worth
shipping) -> SAFETY (guardrail) -> VALIDITY (interaction, fading) -> ECONOMICS (profit at assumed unit costs).
The thresholds are choices, not facts: they are arguments here, never hidden constants.
"""
from __future__ import annotations

import numpy as np

SRM_ALPHA = 0.001          # sample-ratio check: below this the split is broken
ALPHA = 0.05               # family-wise error rate for the Holm correction
MIN_EFFECT = 0.03          # smallest relative lift worth shipping (a business assumption)
TREND_ALPHA = 0.05         # weekly-trend test for a fading effect
INTERACTION_ALPHA = 0.05

DECISIONS = ("DO NOT TRUST", "DO NOT SHIP (harmful)", "SHIP", "ITERATE", "INCONCLUSIVE (underpowered)", "NO EFFECT FOUND")


def decide(r: dict, min_effect: float | None = None, srm_alpha: float = SRM_ALPHA, currency: str = "INR") -> tuple[str, list[str]]:
    """Pure function of one scorecard row. Returns (decision, reasons)."""
    bar = MIN_EFFECT if min_effect is None else min_effect
    if r["srm_p"] < srm_alpha:
        return "DO NOT TRUST", [f"sample ratio mismatch (p = {r['srm_p']:.1e}): arms are {r['n_treat']:,} vs {r['n_ctrl']:,}, "
                                "so assignment is broken and the result cannot be read"]
    notes = []
    if r["srm_p"] < 0.10:
        notes.append(f"sample-ratio p = {r['srm_p']:.3f} is borderline; worth a look at the assignment code")
    if r["significant"] and r["lift"] < 0:
        return "DO NOT SHIP (harmful)", [f"significantly negative lift {r['lift']:+.1%} (Holm-adjusted p = {r['p_adj']:.3g})"]
    if r["significant"]:
        blockers = []
        if r["guardrail_breach"]:
            blockers.append(f"guardrail metric is significantly negative ({r['guardrail_lift']:+.1%})")
        if r["interaction_p"] is not None and r["interaction_p"] < INTERACTION_ALPHA:
            blockers.append(f"overlapping experiment {r['overlap_with']} interacts with it (p = {r['interaction_p']:.3f})")
        if r["trend_p"] is not None and r["trend_slope"] < 0 and r["trend_p"] < TREND_ALPHA:
            blockers.append(f"effect is fading week over week (trend p = {r['trend_p']:.3f}): extend and re-read")
        if r.get("net_ci_high") is not None and not np.isnan(r["net_ci_high"]) and r["net_ci_high"] < 0:
            blockers.append(f"the lift is real but unprofitable at the assumed economics: {currency} {r['net_value']:+.2f} per user "
                            f"(95% CI {r['net_ci_low']:+.2f} to {r['net_ci_high']:+.2f}); do not ship to everyone, target a subset")
        if blockers:
            return "ITERATE", blockers
        notes.insert(0, f"lift {r['lift']:+.1%} (95% CI {r['ci_low']:+.1%} to {r['ci_high']:+.1%}), Holm-adjusted p = {r['p_adj']:.3g}")
        if r.get("net_value") is not None and not np.isnan(r["net_value"]):
            notes.append(f"profit at the assumed economics: {currency} {r['net_value']:+.2f} per user (95% CI {r['net_ci_low']:+.2f} to {r['net_ci_high']:+.2f})")
        if r["ci_low"] < bar:
            notes.append(f"the interval's low end ({r['ci_low']:+.1%}) is below the {bar:.0%} worth-shipping bar")
        if r["overlap_with"] and r["interaction_p"] is not None:
            notes.append(f"overlaps {r['overlap_with']}; no interaction detected (p = {r['interaction_p']:.2f}), which is not proof there is none")
        if r["latest_week_lift"] is not None and r["latest_week_ci_half"] is not None and r["latest_week_lift"] - r["latest_week_ci_half"] <= 0:
            notes.append(f"latest full week lift {r['latest_week_lift']:+.1%} +/- {r['latest_week_ci_half']:.1%} no longer clears zero "
                         "(not a significant fade, but watch it)")
        return "SHIP", notes
    # not significant
    if r["underpowered"]:
        return "INCONCLUSIVE (underpowered)", [f"lift {r['lift']:+.1%} not significant, and this sample cannot detect lifts below "
                                              f"{r['mde']:.1%} (worth-shipping bar is {bar:.0%}): unknown, not zero"] + notes
    return "NO EFFECT FOUND", [f"lift {r['lift']:+.1%} (95% CI {r['ci_low']:+.1%} to {r['ci_high']:+.1%}); the test could have "
                              f"detected {r['mde']:.1%}"] + notes
