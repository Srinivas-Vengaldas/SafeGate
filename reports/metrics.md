Held-out source: `Lakera/gandalf_ignore_instructions`. Threshold 0.472 chosen on the validation split (middle of the best-F1 range).

| Metric | Rules only | Classifier |
| --- | ---: | ---: |
| Test precision | 100.0% | 97.3% |
| Test recall | 2.5% | 91.2% |
| Test F1 | 4.9% | 94.2% |
| Test false-positive rate | 0.0% | 0.3% |
| Held-out source recall | 12.7% | 87.2% |
| Tricky-benign false-positive rate | 0.0% | 4.7% |
| Test PR AUC | – | 0.988 |
| CPU latency per prompt (p50 / p95) | <1 ms | 65.1 / 343.35 ms |

Threshold sweep (classifier only):

| Threshold | Test recall | Test FPR | Held-out recall | Tricky-benign FPR |
| ---: | ---: | ---: | ---: | ---: |
| 0.472 (chosen) | 91.2% | 0.3% | 87.2% | 4.7% |
| 0.500 | 91.2% | 0.3% | 87.0% | 4.7% |
| 0.900 | 90.0% | 0.2% | 83.8% | 2.0% |
| 0.990 | 86.2% | 0.2% | 78.2% | 0.7% |
