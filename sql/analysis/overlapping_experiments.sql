-- Experiments that claim the same surface (exclusive_group) must not share users.
-- Count users enrolled in two experiments of the same group at the same time.
SELECT e1.experiment_id AS exp_a,
       e2.experiment_id AS exp_b,
       e1.exclusive_group,
       COUNT(DISTINCT a1.user_id) AS shared_users
FROM experiments e1
JOIN experiments e2
  ON e1.exclusive_group = e2.exclusive_group
 AND e1.experiment_id < e2.experiment_id
 AND e1.start_date <= e2.end_date AND e2.start_date <= e1.end_date
JOIN assignments a1 ON a1.experiment_id = e1.experiment_id
JOIN assignments a2 ON a2.experiment_id = e2.experiment_id AND a2.user_id = a1.user_id
GROUP BY e1.experiment_id, e2.experiment_id, e1.exclusive_group;
