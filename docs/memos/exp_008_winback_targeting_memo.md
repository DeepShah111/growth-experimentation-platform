# Targeting memo: win-back offer (exp_008)

**Decision needed:** ship the offer to everyone, to a targeted subset, or not at all.   **Date:** 2026-10-18   **Author:** Deep Shah
*All data in this project is simulated (a fictional grocery-delivery app). Profit figures use ASSUMED economics, not measurements.*

## Recommendation
**Do not send the offer to everyone. Send it to the roughly top 24% of users ranked by an S-learner uplift model.** The offer raises orders overall (+7.1%, 95% interval +4.4% to +9.8%), but at the assumed economics (INR 50 per extra order, INR 3 per user shown the offer) showing it to everyone loses money. The main risk is that the profit estimate is rough and rests on those assumptions.

## What we tested
- Change: a win-back offer, shown to a random half of users (50,201 treatment, 49,799 control).
- Primary metric: orders per user. The offer works very differently by user: the lowest-scored tenth of users ordered about 0.16 fewer orders per user when shown it, the highest-scored tenth about 0.24 more.

## Result
| question | answer |
|---|---|
| Overall lift | +7.1% (95% CI +4.4% to +9.8%), significant after Holm correction |
| Profit if shown to everyone | INR -1.29 per user (95% CI -2.03 to -0.55), i.e. about INR -129,000 per 100,000 users |
| Profit if shown to the model's top slice (cut-off chosen on one half, scored on the other) | S-learner INR 1,107 per 1,000 users (INR 1,260 when the cut-off is scored on the same data) |
| Other targeting models, same method | X-learner 944, T-learner 647, random targeting -138 per 1,000 users |

Why "everyone" loses: the overall effect is about +0.034 orders per user, worth about INR 1.71 at INR 50 per order, which is less than the INR 3 cost. Showing the offer to everyone only breaks even if an order is worth about INR 88, or the offer costs under INR 1.71 per user (simple arithmetic on these figures).

## Can we trust it? (checks run)
- [x] Sample-ratio check passed (p = 0.204).
- [x] Cross-fitting: every user's score comes from a model that did not train on that user.
- [x] Seeds: the evaluation was repeated over 10 seeds. The S-learner scored 22.6 extra orders per 1,000 users vs random (sd 1.2 across seeds) and was the best of the three in 10 of 10 seeds.
- [x] Honest profit cut-off: chosen on half of the users, scored on the other half (the same-data figure is shown above to make the optimism visible).
- [~] Learner ranking: S beats T clearly (paired bootstrap +9.3, 95% CI +4.1 to +15.0). S versus X is marginal (+5.1, +0.3 to +10.3) and X versus T cannot be separated. All three beat random targeting by a wide margin.
- [ ] Not tested: a fresh randomised holdout of the targeted group.

## Who benefits?
Measured uplift (extra orders per user) by predicted-uplift tenth for the S-learner: lowest tenth -0.158, then close to zero (at most +0.012) for the next four tenths, rising to +0.158 (ninth tenth) and +0.242 (top tenth). The model's predicted values are close to the measured ones at the top (0.222 predicted vs 0.242 measured).

## Business impact (state assumptions)
Value per incremental order INR 50 (about 12% of a ~INR 400 basket); cost per user shown the offer INR 3. Expected profit of the targeted policy: about INR 1,107 per 1,000 users on held-out users (spread across seeds: sd about INR 368). These are assumptions, not measured margins.

## Risks and what would change my mind
- The unit economics are assumed; if an order is worth about INR 88 or more, or the offer is cheaper than INR 1.71, showing it to everyone becomes defensible.
- Hold-out profit varies a lot between seeds (sd INR 224 to 378 per 1,000 users), so treat the figure as a range, not a point.
- Users with negative predicted uplift (the lowest tenth) are actively harmed by the offer; excluding them matters as much as finding the best responders.

## Next steps
Run a fresh randomised test of the targeted policy against a no-offer holdout before rollout (owner: growth analytics); revisit the INR 50 and INR 3 assumptions with finance.
