# Task 1 Data Contract

Task 1 uses fake/synthetic DESD seed export data. The Advanced AI repo consumes the CSV files as static dataset artifacts and does not import Django models or connect to the DESD database.

Expected location:

```text
data/task1/desd_seed_export/
```

Required files:

```text
customers.csv
producers.csv
products.csv
orders.csv
order_items.csv
README.md
```

Required columns:

```text
customers.csv: customer_id, customer_type, postcode_area
producers.csv: producer_id, producer_name, postcode_area
products.csv: product_id, product_name, category, producer_id, seasonal_start_month, seasonal_end_month, base_price
orders.csv: order_id, customer_id, order_date, total_amount
order_items.csv: order_id, product_id, quantity, unit_price
```

Privacy rule: `customers.csv` must not include names, emails, phone numbers, or full addresses. The recommender only expects anonymised customer IDs.

The CSVs are suitable for proof-of-concept evaluation, not production recommender performance claims.
