1000 requests per target, concurrency 1.

| Target | p50 ms | p95 ms | p99 ms | Added p50 / p95 ms | Requests/s |
|---|---|---|---|---|---|
| direct | 0.63 | 0.83 | 0.99 | 0.0 / 0.0 | 1480.8 |
| no-rails | 2.02 | 2.77 | 14.74 | 1.39 / 1.94 | 368.0 |
| rails | 45.38 | 48.75 | 65.1 | 44.75 / 47.92 | 21.8 |
| cached | 1.94 | 2.27 | 3.83 | 1.31 / 1.44 | 490.2 |
