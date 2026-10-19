-- STAGING: one trustworthy row per real event.
--   * The tracking pipeline double-logs ~0.6% of events (identical session, type, timestamp, product).
--     We keep the lowest event_id of each identical group.
--   * Some events arrive more than 1 day after they happened (late data). We keep them but flag them,
--     because a report run "yesterday" would not have seen them.
CREATE OR REPLACE VIEW stg_events_clean AS
SELECT event_id, session_id, user_id, event_type, event_ts, product_id, ingested_at,
       (ingested_at > event_ts + INTERVAL 1 DAY) AS is_late
FROM (
    SELECT e.*,
           ROW_NUMBER() OVER (PARTITION BY session_id, event_type, event_ts, product_id
                              ORDER BY event_id) AS rn
    FROM events e
)
WHERE rn = 1;
