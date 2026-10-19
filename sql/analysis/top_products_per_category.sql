-- Top-N per group: the 3 most carted products inside each category (DENSE_RANK + QUALIFY).
-- Ties share a rank, so a category can return more than 3 rows when products tie.
WITH carts AS (
    SELECT e.product_id, COUNT(*) AS cart_adds
    FROM stg_events_clean e
    WHERE e.event_type = 'add_to_cart'
    GROUP BY e.product_id
)
SELECT p.category, p.product_id, c.cart_adds,
       DENSE_RANK() OVER (PARTITION BY p.category ORDER BY c.cart_adds DESC) AS rank_in_category
FROM carts c
JOIN products p USING (product_id)
QUALIFY rank_in_category <= 3
ORDER BY p.category, rank_in_category, p.product_id;
