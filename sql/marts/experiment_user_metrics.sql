-- One row per assigned user for ONE experiment ($exp_id).
--   *_in  : outcome, from assignment time to the end of the experiment
--   *_pre : the same user's behaviour in the 28 days BEFORE the experiment started
--           (CUPED covariate now, and uplift-model features later)
WITH a AS (
    SELECT x.experiment_id, x.user_id, x.variant, x.assigned_at,
           e.start_date, e.end_date
    FROM assignments x
    JOIN experiments e ON e.experiment_id = x.experiment_id
    WHERE x.experiment_id = $exp_id
),
ord AS (
    SELECT a.user_id,
           COUNT(*) FILTER (WHERE o.order_ts >= a.assigned_at
                              AND o.order_ts <  a.end_date + INTERVAL 1 DAY)            AS orders_in,
           COALESCE(SUM(o.order_value) FILTER (WHERE o.order_ts >= a.assigned_at
                              AND o.order_ts <  a.end_date + INTERVAL 1 DAY), 0)        AS revenue_in,
           COUNT(*) FILTER (WHERE o.order_ts >= a.start_date - INTERVAL 28 DAY
                              AND o.order_ts <  a.start_date)                           AS orders_pre,
           COALESCE(SUM(o.order_value) FILTER (WHERE o.order_ts >= a.start_date - INTERVAL 28 DAY
                              AND o.order_ts <  a.start_date), 0)                       AS revenue_pre
    FROM a
    JOIN orders o ON o.user_id = a.user_id
    GROUP BY a.user_id
),
ses AS (
    SELECT a.user_id,
           COUNT(*) AS sessions_pre
    FROM a
    JOIN sessions s ON s.user_id = a.user_id
    WHERE s.session_start >= a.start_date - INTERVAL 28 DAY
      AND s.session_start <  a.start_date
    GROUP BY a.user_id
)
SELECT a.user_id, a.variant, u.is_new, u.device, u.channel, u.city,
       DATE_DIFF('day', u.signup_date, a.start_date)  AS days_since_signup,
       COALESCE(o.orders_in, 0)    AS orders_in,
       COALESCE(o.revenue_in, 0)   AS revenue_in,
       COALESCE(o.orders_pre, 0)   AS orders_pre,
       COALESCE(o.revenue_pre, 0)  AS revenue_pre,
       COALESCE(s.sessions_pre, 0) AS sessions_pre
FROM a
JOIN users u ON u.user_id = a.user_id
LEFT JOIN ord o ON o.user_id = a.user_id
LEFT JOIN ses s ON s.user_id = a.user_id
ORDER BY a.user_id;   -- fixed row order: cross-fitting folds and bootstraps must not change between runs
