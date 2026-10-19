-- INTERACTION table for two experiments ($exp_a, $exp_b): users enrolled in both, split into the 2x2 grid
-- of (variant in A) x (variant in B), with orders during the period where both were running.
WITH ea AS (SELECT CAST(start_date AS TIMESTAMP) AS s, CAST(end_date AS TIMESTAMP) + INTERVAL 1 DAY AS e
            FROM experiments WHERE experiment_id = $exp_a),
     eb AS (SELECT CAST(start_date AS TIMESTAMP) AS s, CAST(end_date AS TIMESTAMP) + INTERVAL 1 DAY AS e
            FROM experiments WHERE experiment_id = $exp_b),
     win AS (SELECT GREATEST(ea.s, eb.s) AS w_start, LEAST(ea.e, eb.e) AS w_end FROM ea, eb),
     both_users AS (
         SELECT a.user_id, a.variant AS variant_a, b.variant AS variant_b,
                GREATEST(a.assigned_at, b.assigned_at) AS enrolled_at
         FROM assignments a
         JOIN assignments b ON b.user_id = a.user_id AND b.experiment_id = $exp_b
         WHERE a.experiment_id = $exp_a
     ),
     per_user AS (
         SELECT u.user_id, u.variant_a, u.variant_b,
                COUNT(o.order_id) AS n_orders
         FROM both_users u CROSS JOIN win
         LEFT JOIN orders o ON o.user_id = u.user_id
                           AND o.order_ts >= GREATEST(u.enrolled_at, win.w_start)
                           AND o.order_ts <  win.w_end
         GROUP BY u.user_id, u.variant_a, u.variant_b
     )
SELECT variant_a, variant_b,
       COUNT(*)                         AS users,
       SUM(n_orders)                    AS orders,
       SUM(n_orders * n_orders)         AS orders_sq,       -- lets the caller compute a variance
       ROUND(AVG(n_orders), 5)          AS orders_per_user
FROM per_user
GROUP BY variant_a, variant_b
ORDER BY variant_a, variant_b;
