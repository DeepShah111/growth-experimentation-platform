-- SESSIONIZATION from raw events with LAG(): a new session starts when the gap to the user's previous
-- event is more than 30 minutes. A running SUM() of those "new session" flags numbers the sessions.
-- Used as an independent check on the logged session_id: how many sessions would we rebuild?
WITH ordered AS (
    SELECT user_id, event_ts,
           LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts, event_id) AS prev_ts
    FROM stg_events_clean
),
flagged AS (
    SELECT user_id, event_ts,
           CASE WHEN prev_ts IS NULL OR event_ts > prev_ts + INTERVAL 30 MINUTE THEN 1 ELSE 0 END AS new_session
    FROM ordered
),
rebuilt AS (
    SELECT user_id, SUM(new_session) AS rebuilt_sessions
    FROM flagged
    GROUP BY user_id
),
logged AS (
    SELECT user_id, COUNT(*) AS logged_sessions
    FROM sessions
    GROUP BY user_id
)
SELECT u.device,
       SUM(l.logged_sessions)                                          AS logged_sessions,
       SUM(r.rebuilt_sessions)                                         AS rebuilt_sessions,
       ROUND(SUM(r.rebuilt_sessions) * 1.0 / SUM(l.logged_sessions), 4) AS rebuilt_to_logged_ratio
FROM users u
JOIN logged l USING (user_id)
JOIN rebuilt r USING (user_id)
GROUP BY u.device
ORDER BY u.device;
