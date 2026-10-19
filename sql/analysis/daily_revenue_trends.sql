-- Window-function showcase on a daily series:
--   running total (SUM OVER), 7-day moving average (ROWS frame), day-over-day and week-over-week change (LAG).
WITH daily AS (
    SELECT CAST(order_ts AS DATE) AS day,
           COUNT(*)               AS orders,
           SUM(order_value)       AS revenue
    FROM orders
    GROUP BY 1
)
SELECT day,
       orders,
       ROUND(revenue, 2)                                                                       AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)                                              AS cumulative_revenue,
       ROUND(AVG(revenue) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2)     AS revenue_7d_avg,
       ROUND(revenue / NULLIF(LAG(revenue, 1) OVER (ORDER BY day), 0) - 1, 4)                  AS dod_change,
       ROUND(revenue / NULLIF(LAG(revenue, 7) OVER (ORDER BY day), 0) - 1, 4)                  AS wow_change
FROM daily
ORDER BY day;
