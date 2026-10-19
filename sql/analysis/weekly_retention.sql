-- Weekly retention for users who signed up DURING the simulation window.
-- Cohort = signup week. Retained in week k = had a session k weeks after signup week.
WITH cohort AS (
    SELECT user_id, DATE_TRUNC('week', signup_date) AS cohort_week
    FROM users
    WHERE is_new
),
activity AS (
    SELECT DISTINCT user_id, DATE_TRUNC('week', session_start) AS active_week
    FROM sessions
),
joined AS (
    SELECT c.cohort_week,
           DATE_DIFF('week', c.cohort_week, a.active_week) AS weeks_since_signup,
           c.user_id
    FROM cohort c
    JOIN activity a USING (user_id)
    WHERE a.active_week >= c.cohort_week
),
sizes AS (
    SELECT cohort_week, COUNT(*) AS cohort_size FROM cohort GROUP BY cohort_week
)
SELECT j.cohort_week,
       s.cohort_size,
       j.weeks_since_signup,
       COUNT(DISTINCT j.user_id)                                  AS active_users,
       ROUND(COUNT(DISTINCT j.user_id) * 1.0 / s.cohort_size, 3)  AS retention
FROM joined j
JOIN sizes s USING (cohort_week)
WHERE j.weeks_since_signup BETWEEN 0 AND 8
GROUP BY j.cohort_week, s.cohort_size, j.weeks_since_signup
ORDER BY j.cohort_week, j.weeks_since_signup;
