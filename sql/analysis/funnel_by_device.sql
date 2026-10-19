-- Session-level purchase funnel by device, on de-duplicated events (see staging/stg_events_clean.sql).
-- One row per session with a flag per funnel step, then conversion between steps per device.
WITH session_flags AS (
    SELECT session_id,
           MAX((event_type = 'product_view')::INT)   AS viewed,
           MAX((event_type = 'add_to_cart')::INT)    AS carted,
           MAX((event_type = 'checkout_start')::INT) AS checkout,
           MAX((event_type = 'purchase')::INT)       AS purchased
    FROM stg_events_clean
    GROUP BY session_id
)
SELECT s.device,
       COUNT(*)                                                AS sessions,
       ROUND(AVG(f.carted), 4)                                 AS cart_rate,
       ROUND(SUM(f.checkout) / NULLIF(SUM(f.carted), 0), 4)    AS cart_to_checkout,
       ROUND(SUM(f.purchased) / NULLIF(SUM(f.checkout), 0), 4) AS checkout_to_purchase,
       ROUND(AVG(f.purchased), 4)                              AS session_conversion
FROM sessions s
JOIN session_flags f USING (session_id)
GROUP BY s.device
ORDER BY sessions DESC, s.device;
