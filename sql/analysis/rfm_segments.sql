-- RFM customer segmentation in pure SQL, as of $as_of (a DATE).
--   Recency   = days since last order   (lower is better)
--   Frequency = number of orders
--   Monetary  = total order value
-- Each is scored 1-5 with NTILE (5 = best); segment labels come from simple, readable rules.
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
           NTILE(5) OVER (ORDER BY recency_days DESC, user_id) AS r_score,   -- most recent buyers get 5
           NTILE(5) OVER (ORDER BY frequency,        user_id)  AS f_score,
           NTILE(5) OVER (ORDER BY monetary,         user_id)  AS m_score
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
