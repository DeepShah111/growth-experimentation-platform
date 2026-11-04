-- A WHOLE EXPERIMENT ANALYSIS IN SQL: arm sizes (for the sample-ratio check), CUPED adjustment and the lift with its interval.
-- Expects a CTE called `metrics` with one row per user: variant ('control'/'treatment'), orders_in (outcome), orders_pre (covariate).
-- That is exactly what sql/marts/experiment_user_metrics.sql returns; the app wraps it as  WITH metrics AS (<mart>) <this query>.
--   theta  = COVAR_SAMP(y, x) / VAR_SAMP(x)               (pooled over both arms, as in CUPED)
--   y_adj  = y - theta * (x - AVG(x))                      (same mean, smaller variance)
--   lift   = mean_treat / mean_ctrl - 1, interval by the delta method
-- The Python engine (src.engine.analyze) must give the same numbers: tests/sql checks that to 1e-9.
theta AS (
    SELECT COVAR_SAMP(orders_in, orders_pre) / NULLIF(VAR_SAMP(orders_pre), 0) AS th,
           AVG(orders_pre)                                                       AS mx
    FROM metrics
),
adj AS (
    SELECT m.variant,
           m.orders_in                                       AS y_raw,
           m.orders_in - COALESCE(t.th, 0) * (m.orders_pre - t.mx) AS y_adj
    FROM metrics m CROSS JOIN theta t
),
arms AS (
    SELECT variant,
           COUNT(*)           AS n,
           AVG(y_raw)         AS mean_raw,
           VAR_SAMP(y_raw)    AS var_raw,
           AVG(y_adj)         AS mean_adj,
           VAR_SAMP(y_adj)    AS var_adj
    FROM adj
    GROUP BY variant
)
SELECT t.n                                                                   AS n_treat,
       c.n                                                                   AS n_ctrl,
       t.mean_raw / c.mean_raw - 1                                           AS naive_lift,
       t.mean_adj / c.mean_adj - 1                                           AS cuped_lift,
       1.96 * SQRT(t.var_adj / t.n / POWER(c.mean_adj, 2)
                   + POWER(t.mean_adj, 2) * c.var_adj / c.n / POWER(c.mean_adj, 4)) AS cuped_half_width,
       1.96 * SQRT(t.var_raw / t.n / POWER(c.mean_raw, 2)
                   + POWER(t.mean_raw, 2) * c.var_raw / c.n / POWER(c.mean_raw, 4)) AS naive_half_width,
       SQRT(t.var_adj / t.n + c.var_adj / c.n)                               AS se_abs_adjusted
FROM arms t
JOIN arms c ON t.variant = 'treatment' AND c.variant = 'control';
