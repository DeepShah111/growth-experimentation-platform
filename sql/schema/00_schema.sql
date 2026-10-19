-- Build the warehouse from the Parquet files. {dir} is replaced by build_warehouse.py.
-- Observable tables live in the default schema. The hidden truth lives in schema `answer_key`
-- so analysis queries cannot touch it by accident.
CREATE SCHEMA IF NOT EXISTS answer_key;

CREATE OR REPLACE TABLE users       AS SELECT * FROM read_parquet('{dir}/users.parquet');
CREATE OR REPLACE TABLE products    AS SELECT * FROM read_parquet('{dir}/products.parquet');
CREATE OR REPLACE TABLE sessions    AS SELECT * FROM read_parquet('{dir}/sessions.parquet');
CREATE OR REPLACE TABLE events      AS SELECT * FROM read_parquet('{dir}/events.parquet');
CREATE OR REPLACE TABLE orders      AS SELECT * FROM read_parquet('{dir}/orders.parquet');
CREATE OR REPLACE TABLE experiments AS SELECT * FROM read_parquet('{dir}/experiments.parquet');
CREATE OR REPLACE TABLE assignments AS SELECT * FROM read_parquet('{dir}/assignments.parquet');
CREATE OR REPLACE TABLE answer_key.ground_truth AS SELECT * FROM read_parquet('{dir}/answer_key.parquet');
CREATE OR REPLACE TABLE answer_key.user_truth   AS SELECT * FROM read_parquet('{dir}/user_truth.parquet');
