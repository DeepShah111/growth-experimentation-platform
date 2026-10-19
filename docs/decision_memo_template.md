# Decision memo: <experiment name>

**Decision needed:** ship / do not ship / run longer / target a subset.   **Date:** <date>   **Author:** <you>

## Recommendation (2 sentences)
<One clear sentence on what to do. One sentence on the main risk.>

## What we tested
- Change: <what users saw>      - Users: <n per arm>      - Window: <dates>
- Primary metric: <orders per user / revenue per user>, guardrails: <list>

## Result
| metric | control | treatment | relative lift | 95% CI | p-value |
|---|---|---|---|---|---|
| <primary> | | | | | |

## Can we trust it? (checks run)
- [ ] Sample-ratio check passed (observed split vs planned): p = <>
- [ ] Covariate balance / randomisation sane
- [ ] Planned sample size reached (power analysis: detectable lift = <>)
- [ ] No early stopping or correction applied: <how>
- [ ] Multiple experiments in the same period: correction used = <Holm / BH / none> and why
- [ ] Novelty effect checked (lift by week): <stable / decaying>
- [ ] Data quality issues found and handled: <duplicates, late events, overlapping tests>

## Who benefits? (only if segments were pre-registered)
<uplift by segment, with intervals; say plainly if the evidence is weak>

## Business impact (state assumptions)
Value per incremental order = <> ; cost per treated user = <> ; expected profit = <> (assumptions, not measured)

## Risks and what would change my mind
<2-3 bullets>

## Next steps
<owner, date>
