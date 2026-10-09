Held-out source: `Lakera/gandalf_ignore_instructions`. Test split: 682 prompts; held-out: 999 attacks; tricky-benign: 150 safe prompts. Every model is scored as served (max 256 tokens per window, at most 4 windows per prompt), on the same CPU.

| Detector | Threshold | Test precision | Test recall | Test F1 | Test FPR | Held-out recall | Tricky-benign FPR | Latency p50 / p95 | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules only | – | 100.0% | 2.5% | 4.9% | 0.0% | 12.7% | 0.0% | <1 ms | – |
| SafeGate (DeBERTa-v3-small, int8) | 0.975 | 100.0% | 88.8% | 94.0% | 0.0% | 83.8% | 3.3% | 23.54 / 126.56 ms | 176 MB |
| deberta-v3-base-prompt-injection-v2 (int8, default threshold) | 0.500 | 81.1% | 53.8% | 64.7% | 1.7% | 0.0% | 0.0% | 47.21 / 244.46 ms | 248 MB |
| deberta-v3-base-prompt-injection-v2 (int8, threshold tuned on our validation split) | 0.218 | 68.2% | 75.0% | 71.4% | 4.7% | 0.3% | 0.0% | 47.04 / 247.51 ms | 248 MB |
