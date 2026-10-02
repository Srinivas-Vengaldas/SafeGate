Held-out source: `Lakera/gandalf_ignore_instructions`. Threshold 0.016 chosen on the validation split (middle of the best-F1 range).

| Metric | Rules only | Classifier |
| --- | ---: | ---: |
| Test precision | 100.0% | 97.6% |
| Test recall | 1.2% | 100.0% |
| Test F1 | 2.5% | 98.8% |
| Test false-positive rate | 0.0% | 0.6% |
| Held-out source recall | 12.7% | 91.4% |
| Tricky-benign false-positive rate | 0.0% | 18.7% |
| Test PR AUC | – | 1.000 |
| CPU latency per prompt (p50 / p95) | <1 ms | 71.49 / 378.77 ms |

Threshold sweep (classifier only):

| Threshold | Test recall | Test FPR | Held-out recall | Tricky-benign FPR |
| ---: | ---: | ---: | ---: | ---: |
| 0.016 (chosen) | 100.0% | 0.6% | 91.4% | 18.7% |
| 0.500 | 97.5% | 0.0% | 89.1% | 17.3% |
| 0.900 | 93.8% | 0.0% | 87.8% | 12.7% |
| 0.990 | 88.8% | 0.0% | 85.9% | 10.0% |
