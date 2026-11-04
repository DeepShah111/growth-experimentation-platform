-- Weekly retention for users who signed up DURING the simulation window.
-- Cohort = signup week. Retained in week k = had a session k weeks after signup week.
-- RIGHT-CENSORING: a cohort can only be judged at week k if that whole week has already happened. The grid below keeps
-- (cohort, k) pairs whose week k ends before the last (possibly partial) week of data, and counts zero active users
-- explicitly, so late cohorts and the unfinished final week never drag the later-week retention down.
WITH cohort AS (
    SELECT user_id, CAST(DATE_TRUNC('week', signup_date) AS DATE) AS cohort_week
    FROM users
    WHERE is_new
),
bounds AS (
    SELECT CAST(DATE_TRUNC('week', MAX(session_start)) AS DATE) AS last_week FROM sessions
),
sizes AS (
    SELECT cohort_week, COUNT(*) AS cohort_size FROM cohort GROUP BY cohort_week
),
grid AS (
    SELECT s.cohort_week, s.cohort_size, t.k AS weeks_since_signup
    FROM sizes s
    CROSS JOIN range(0, 9) t(k)
    CROSS JOIN bounds b
    WHERE s.cohort_week + t.k * INTERVAL 7 DAY < b.last_week
),
activity AS (
    SELECT DISTINCT user_id, CAST(DATE_TRUNC('week', session_start) AS DATE) AS active_week
    FROM sessions
),
joined AS (
    SELECT c.cohort_week,
           DATE_DIFF('week', c.cohort_week, a.active_week) AS weeks_since_signup,
           c.user_id
    FROM cohort c
    JOIN activity a USING (user_id)
    WHERE a.active_week >= c.cohort_week
)
SELECT g.cohort_week,
       g.cohort_size,
       g.weeks_since_signup,
       COUNT(DISTINCT j.user_id)                                  AS active_users,
       ROUND(COUNT(DISTINCT j.user_id) * 1.0 / g.cohort_size, 3)  AS retention
FROM grid g
LEFT JOIN joined j
       ON j.cohort_week = g.cohort_week AND j.weeks_since_signup = g.weeks_since_signup
GROUP BY g.cohort_week, g.cohort_size, g.weeks_since_signup
ORDER BY g.cohort_week, g.weeks_since_signup;
