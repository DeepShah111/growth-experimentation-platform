-- One row per (user, exposure week) for experiment $exp_id.
-- Exposure week k = days [7(k-1), 7k) after the USER'S OWN assignment time, so every user's novelty
-- clock starts when they first saw the change. Only COMPLETE weeks inside the experiment window are kept,
-- so each row covers exactly 7 days of exposure and rows are comparable across weeks.
WITH ex AS (
    SELECT CAST(end_date AS TIMESTAMP) + INTERVAL 1 DAY AS end_ts
    FROM experiments WHERE experiment_id = $exp_id
),
wk AS (
    SELECT a.user_id, a.variant, a.assigned_at, t.k
    FROM assignments a
    CROSS JOIN ex
    CROSS JOIN range(0, 12) t(k)
    WHERE a.experiment_id = $exp_id
      AND a.assigned_at + (t.k + 1) * INTERVAL 7 DAY <= ex.end_ts
)
SELECT wk.k + 1 AS exposure_week,
       wk.user_id,
       wk.variant,
       COUNT(o.order_id) AS n_orders
FROM wk
LEFT JOIN orders o
       ON o.user_id = wk.user_id
      AND o.order_ts >= wk.assigned_at + wk.k * INTERVAL 7 DAY
      AND o.order_ts <  wk.assigned_at + (wk.k + 1) * INTERVAL 7 DAY
GROUP BY wk.k, wk.user_id, wk.variant
ORDER BY exposure_week, wk.user_id;
