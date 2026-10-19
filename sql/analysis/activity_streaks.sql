-- GAPS AND ISLANDS: longest run of consecutive active days per user.
-- Trick: for consecutive days, (day number - row number) is constant, so it labels each "island".
WITH active_days AS (
    SELECT DISTINCT user_id, CAST(session_start AS DATE) AS day
    FROM sessions
),
numbered AS (
    SELECT user_id, day,
           DATE_DIFF('day', DATE '2000-01-01', day)
             - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY day) AS island
    FROM active_days
),
streaks AS (
    SELECT user_id, island, COUNT(*) AS streak_days
    FROM numbered
    GROUP BY user_id, island
),
longest AS (
    SELECT user_id, MAX(streak_days) AS longest_streak
    FROM streaks
    GROUP BY user_id
)
SELECT longest_streak, COUNT(*) AS users
FROM longest
GROUP BY longest_streak
ORDER BY longest_streak;
