# Decision memo: checkout_button_v2 (exp_001)

**Decision needed:** ship / do not ship.   **Date:** 2026-10-18   **Author:** Deep Shah
*All data in this project is simulated (a fictional grocery-delivery app). The memo shows the format and the reasoning, not a real business result.*

## Recommendation
**Ship checkout_button_v2 to all users.** Orders per user rose by about 11% (95% interval +8.0% to +14.3%), the revenue guardrail rose with it, and all trust checks passed. The main risk is the overlap with homepage_redesign (exp_004), which we could not rule out as an interaction, so keep a 5% holdback for two weeks after rollout.

## What we tested
- Change: a redesigned checkout button, shown to a random half of users.
- Users: 45,034 treatment, 44,893 control. Window: 28 days from experiment start (days 28 to 55).
- Primary metric: orders per user (variance-reduced with CUPED using the 28-day pre-period). Guardrail: revenue per user.

## Result
| metric | relative lift (CUPED) | 95% CI | Holm-adjusted p |
|---|---|---|---|
| orders per user (primary) | +11.1% | +8.0% to +14.3% | 4.4e-12 |
| revenue per user (guardrail) | +11.1% | not computed | n/a |

For forecasting, plan on the low end of the interval (about +8%), not the +11.1% point estimate: a single experiment tends to overstate its own winner.

## Can we trust it? (checks run)
- [x] Sample-ratio check passed: 44,893 vs 45,034 users, p = 0.638 (a failure would be p below 0.001).
- [ ] Covariate balance: not run separately for this experiment (only the sample-ratio check).
- [x] Power: the experiment could detect a lift of about 4.1% or more; the observed lift is well above that.
- [x] No early stopping: analysed once, on the complete window.
- [x] Multiple experiments: Holm correction across the 8 experiments that passed the sample-ratio check (exp_006 was excluded as untrustworthy).
- [x] Novelty effect checked: the weekly trend is flat to slightly rising (+0.0017 orders per user-week per week, p = 0.25); the latest full week is +14.9% (+/- 6.0).
- [x] Data quality handled: 0.60% duplicate events removed, 1.51% of events arrived more than a day late (kept, flagged).
- [~] Overlap: exp_001 and exp_004 both claim the homepage surface and share users. Effect of exp_001 is +0.0414 orders per user when exp_004 is on and +0.0288 when it is off; the interaction is +0.0126 +/- 0.0203 (p = 0.22). No interaction detected, which is not proof there is none.

## Who benefits?
Not analysed: segments were not pre-registered for this test.

## Business impact (state assumptions)
At an assumed INR 50 contribution per order, +0.0288 to +0.0414 extra orders per user is about INR 1.4 to 2.1 per user over the 28-day window. This is an assumption, not a measurement.

## Risks and what would change my mind
- A significant interaction with exp_004 in the holdback data would mean the +11% depends on the homepage change that may not ship.
- A fade in weeks 5 to 8 (our window only covers four weeks).
- The estimate comes from one experiment on simulated data; a replication would narrow the interval.

## Next steps
Roll out to 95% with a 5% holdback; re-read at 2 weeks (owner: growth analytics).
