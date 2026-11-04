-- RFM customer segmentation in pure SQL, as of $as_of (a DATE).
--   Recency   = days since last order   (lower is better)
--   Frequency = number of orders
--   Monetary  = total order value
-- Each is scored 1-5 by RANK position (5 = best); segment labels come from simple, readable rules.
-- RANK, not NTILE: users with identical values (most buyers have 1 order) must get the SAME score. NTILE would split
-- such ties across different scores by user_id, which is arbitrary. With ties, the lowest scores can be empty on purpose.
WITH base AS (
    SELECT user_id,
           DATE_DIFF('day', CAST(MAX(order_ts) AS DATE), CAST($as_of AS DATE)) AS recency_days,
           COUNT(*)                                                            AS frequency,
           SUM(order_value)                                                    AS monetary
    FROM orders
    WHERE CAST(order_ts AS DATE) <= CAST($as_of AS DATE)
    GROUP BY user_id
),
scored AS (
    SELECT *,
           1 + LEAST(4, FLOOR(5.0 * (RANK() OVER (ORDER BY recency_days DESC) - 1) / COUNT(*) OVER ()))::INT AS r_score,  -- most recent buyers get 5
           1 + LEAST(4, FLOOR(5.0 * (RANK() OVER (ORDER BY frequency)        - 1) / COUNT(*) OVER ()))::INT AS f_score,
           1 + LEAST(4, FLOOR(5.0 * (RANK() OVER (ORDER BY monetary)         - 1) / COUNT(*) OVER ()))::INT AS m_score
    FROM base
),
labelled AS (
    SELECT *,
           CASE WHEN r_score >= 4 AND f_score >= 4 THEN 'champions'
                WHEN r_score >= 3 AND f_score >= 3 THEN 'loyal'
                WHEN r_score >= 4 AND f_score <= 2 THEN 'new_or_promising'
                WHEN r_score <= 2 AND f_score >= 3 THEN 'at_risk'
                WHEN r_score <= 2 AND f_score <= 2 THEN 'lost'
                ELSE 'needs_attention' END AS segment
    FROM scored
)
SELECT segment,
       COUNT(*)                          AS users,
       ROUND(AVG(recency_days), 1)       AS avg_recency_days,
       ROUND(AVG(frequency), 2)          AS avg_orders,
       ROUND(AVG(monetary), 2)           AS avg_revenue,
       ROUND(SUM(monetary) * 1.0 / SUM(SUM(monetary)) OVER (), 4) AS share_of_revenue
FROM labelled
GROUP BY segment
ORDER BY share_of_revenue DESC, segment;
