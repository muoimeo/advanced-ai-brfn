# Task 1 Limitations

Task 1 is a transparent quick-reorder proof of concept using fake/synthetic DESD-style data.

Current DESD seed export scale:

```text
customers: 80
producers: 12
products: 60
orders: 483
order_lines: 1,794
date_range: 2026-01-10 to 2026-05-01
```

Key limitations:

- The data is not real BRFN customer behaviour.
- Metrics such as Precision@3, Recall@3 and HitRate@3 should not be read as production accuracy.
- Customer segment patterns are generated/seeded assumptions.
- The recommender does not learn long-term behavioural change from real marketplace logs.
- Producer fairness is measured through coverage and recommendation share, but no automatic fairness re-ranking is applied in v1.
- Cold-start customers fall back to global popularity, which can over-promote already popular products.
- Producer demand trends and next-week forecast outputs are descriptive aggregates, not a production forecasting model.
- The "You may also like" branch is a discovery enhancement, not the core quick-reorder requirement.
- Discovery is evaluated against future unseen synthetic purchases, not real customer discovery behaviour.
- Producer diversity is applied as a re-ranking preference, not a hard guarantee, to avoid severely reducing recommendation relevance.

The value of this implementation is the data contract, reusable recommender pipeline, temporal evaluation method, reason codes, and API interface that can later consume real DESD order history.
