Held-out source: `Lakera/gandalf_ignore_instructions`. Threshold 0.004 chosen on the validation split (middle of the best-F1 range).

| Metric | Rules only | Classifier |
| --- | ---: | ---: |
| Test precision | 33.3% | 69.8% |
| Test recall | 0.9% | 84.5% |
| Test F1 | 1.7% | 76.4% |
| Test false-positive rate | 0.5% | 11.2% |
| Test F1, prompts only | 4.9% | 87.2% |
| Test recall, document passages | 0.0% | 77.0% |
| Test false-positive rate, document passages | 2.5% | 38.8% |
| Held-out source recall | 12.7% | 97.9% |
| Tricky-benign false-positive rate | 0.0% | 19.3% |
| Test PR AUC | – | 0.884 |
| CPU latency per prompt (p50 / p95) | <1 ms | 58.69 / 117.33 ms |

Threshold sweep (classifier only):

| Threshold | Test recall | Test FPR | Held-out recall | Tricky-benign FPR |
| ---: | ---: | ---: | ---: | ---: |
| 0.004 (chosen) | 84.0% | 11.2% | 97.8% | 19.3% |
| 0.500 | 56.0% | 1.6% | 89.8% | 5.3% |
| 0.900 | 52.2% | 0.9% | 86.6% | 4.7% |
| 0.990 | 48.3% | 0.3% | 83.0% | 3.3% |
