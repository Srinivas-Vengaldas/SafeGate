Held-out source: `Lakera/gandalf_ignore_instructions`. Threshold 0.002 chosen on the validation split (middle of the best-F1 range).

| Metric | Rules only | Classifier |
| --- | ---: | ---: |
| Test precision | 33.3% | 78.9% |
| Test recall | 0.9% | 83.6% |
| Test F1 | 1.7% | 81.2% |
| Test false-positive rate | 0.5% | 6.8% |
| Test F1, prompts only | 4.9% | 88.6% |
| Test recall, document passages | 0.0% | 76.3% |
| Test false-positive rate, document passages | 2.5% | 21.2% |
| Held-out source recall | 12.7% | 98.1% |
| Tricky-benign false-positive rate | 0.0% | 18.7% |
| Test PR AUC | – | 0.906 |
| CPU latency per prompt (p50 / p95) | <1 ms | 92.49 / 200.48 ms |

Threshold sweep (classifier only):

| Threshold | Test recall | Test FPR | Held-out recall | Tricky-benign FPR |
| ---: | ---: | ---: | ---: | ---: |
| 0.002 (chosen) | 84.5% | 7.1% | 98.2% | 18.7% |
| 0.500 | 50.9% | 0.3% | 87.0% | 4.7% |
| 0.900 | 48.3% | 0.1% | 83.8% | 2.0% |
| 0.990 | 45.3% | 0.0% | 78.2% | 0.7% |
