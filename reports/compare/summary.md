## Injection detectors on the same test sets

Held-out source: `Lakera/gandalf_ignore_instructions`. Test split: 994 prompts; held-out: 999 attacks; tricky-benign: 150 safe prompts. Every model is scored as served (max 256 tokens per window, at most 4 windows per prompt), on the same CPU.

| Detector | Threshold | Test precision | Test recall | Test F1 | Test FPR | Held-out recall | Tricky-benign FPR | Latency p50 / p95 | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules only | – | 33.3% | 0.9% | 1.7% | 0.5% | 12.7% | 0.0% | <1 ms | – |
| SafeGate (DeBERTa-v3-small, int8) | 0.007 | 98.7% | 97.8% | 98.3% | 0.4% | 85.3% | 5.3% | 56.95 / 114.89 ms | 176 MB |
| deberta-v3-base-prompt-injection-v2 (as published, PyTorch fp32) | 0.500 | 88.5% | 46.6% | 61.0% | 1.8% | 100.0% | 6.7% | 176.74 / 368.7 ms | – |
| deberta-v3-base-prompt-injection-v2 (fp32, threshold tuned on our validation split) | 0.000 | 74.0% | 73.7% | 73.9% | 7.9% | 100.0% | 12.0% | 176.43 / 368.92 ms | – |
| deberta-v3-base-prompt-injection-v2 (ONNX int8, like SafeGate) | 0.500 | 100.0% | 0.4% | 0.9% | 0.0% | 0.0% | 0.0% | 111.08 / 217.83 ms | 248 MB |

Training datasets named in the baseline's model card that we also use:
- jackhhao/jailbreak-classification
