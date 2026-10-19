-- Sample Ratio Mismatch input: users per arm for every experiment. Planned split is 50/50.
SELECT experiment_id,
       SUM((variant = 'control')::INT)   AS control_users,
       SUM((variant = 'treatment')::INT) AS treatment_users
FROM assignments
GROUP BY experiment_id
ORDER BY experiment_id;
