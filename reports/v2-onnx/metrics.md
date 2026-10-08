Held-out source: `Lakera/gandalf_ignore_instructions`. Threshold 0.975 chosen on the validation split (middle of the best-F1 range).

| Metric | Rules only | Classifier |
| --- | ---: | ---: |
| Test precision | 100.0% | 98.6% |
| Test recall | 2.5% | 88.8% |
| Test F1 | 4.9% | 93.4% |
| Test false-positive rate | 0.0% | 0.2% |
| Held-out source recall | 12.7% | 83.8% |
| Tricky-benign false-positive rate | 0.0% | 3.3% |
| Test PR AUC | – | 0.987 |
| CPU latency per prompt (p50 / p95) | <1 ms | 43.29 / 209.04 ms |

Threshold sweep (classifier only):

| Threshold | Test recall | Test FPR | Held-out recall | Tricky-benign FPR |
| ---: | ---: | ---: | ---: | ---: |
| 0.500 | 91.2% | 0.3% | 89.9% | 5.3% |
| 0.900 | 91.2% | 0.3% | 85.4% | 4.0% |
| 0.975 (chosen) | 88.8% | 0.2% | 83.8% | 3.3% |
| 0.990 | 88.8% | 0.2% | 81.8% | 2.7% |
