-- DATA-QUALITY MONITOR for the event stream, per day:
--   duplicate_rate : share of raw events that are exact repeats of an earlier event
--   late_rate      : share of distinct events that arrived more than 1 day after they happened
WITH raw AS (
    SELECT CAST(event_ts AS DATE) AS day, COUNT(*) AS raw_events FROM events GROUP BY 1
),
clean AS (
    SELECT CAST(event_ts AS DATE) AS day,
           COUNT(*)                     AS distinct_events,
           SUM(is_late::INT)            AS late_events
    FROM stg_events_clean
    GROUP BY 1
)
SELECT r.day, r.raw_events, c.distinct_events,
       r.raw_events - c.distinct_events                                   AS duplicates_removed,
       ROUND((r.raw_events - c.distinct_events) * 1.0 / r.raw_events, 4)   AS duplicate_rate,
       c.late_events,
       ROUND(c.late_events * 1.0 / c.distinct_events, 4)                   AS late_rate
FROM raw r
JOIN clean c USING (day)
ORDER BY r.day;
