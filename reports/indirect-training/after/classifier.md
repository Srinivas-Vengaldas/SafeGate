Held-out source: `Lakera/gandalf_ignore_instructions`. Threshold 0.007 chosen on the validation split (middle of the best-F1 range).

| Metric | Rules only | Classifier |
| --- | ---: | ---: |
| Test precision | 33.3% | 98.3% |
| Test recall | 0.9% | 97.4% |
| Test F1 | 1.7% | 97.8% |
| Test false-positive rate | 0.5% | 0.5% |
| Test F1, prompts only | 4.9% | 96.2% |
| Test recall, document passages | 0.0% | 98.0% |
| Test false-positive rate, document passages | 2.5% | 0.0% |
| Held-out source recall | 12.7% | 83.4% |
| Tricky-benign false-positive rate | 0.0% | 4.7% |
| Test PR AUC | – | 0.994 |
| CPU latency per prompt (p50 / p95) | <1 ms | 43.92 / 80.7 ms |

Threshold sweep (classifier only):

| Threshold | Test recall | Test FPR | Held-out recall | Tricky-benign FPR |
| ---: | ---: | ---: | ---: | ---: |
| 0.007 (chosen) | 97.4% | 0.5% | 83.4% | 4.7% |
| 0.500 | 92.2% | 0.0% | 69.0% | 1.3% |
| 0.900 | 87.1% | 0.0% | 64.0% | 1.3% |
| 0.990 | 74.6% | 0.0% | 55.7% | 0.7% |
