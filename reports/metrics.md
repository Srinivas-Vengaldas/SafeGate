Held-out source: `Lakera/gandalf_ignore_instructions`. Threshold 0.015 chosen on the validation split (middle of the best-F1 range).

| Metric | Rules only | Classifier |
| --- | ---: | ---: |
| Test precision | 33.3% | 99.6% |
| Test recall | 0.9% | 96.5% |
| Test F1 | 1.7% | 98.0% |
| Test false-positive rate | 0.5% | 0.1% |
| Test F1, prompts only | 4.9% | 96.2% |
| Test recall, document passages | 0.0% | 98.0% |
| Test false-positive rate, document passages | 2.5% | 0.0% |
| Held-out source recall | 12.7% | 76.1% |
| Tricky-benign false-positive rate | 0.0% | 2.7% |
| Test PR AUC | – | 0.996 |
| CPU latency per prompt (p50 / p95) | <1 ms | 67.72 / 115.41 ms |

Threshold sweep (classifier only):

| Threshold | Test recall | Test FPR | Held-out recall | Tricky-benign FPR |
| ---: | ---: | ---: | ---: | ---: |
| 0.015 (chosen) | 96.5% | 0.1% | 76.1% | 2.7% |
| 0.500 | 95.3% | 0.0% | 68.0% | 0.7% |
| 0.900 | 94.4% | 0.0% | 64.9% | 0.0% |
| 0.990 | 93.5% | 0.0% | 60.4% | 0.0% |
