## Injection detectors on the same test sets

Held-out source: `Lakera/gandalf_ignore_instructions`. Test split: 682 prompts; held-out: 999 attacks; tricky-benign: 150 safe prompts. Every model is scored as served (max 256 tokens per window, at most 4 windows per prompt), on the same CPU.

| Detector | Threshold | Test precision | Test recall | Test F1 | Test FPR | Held-out recall | Tricky-benign FPR | Latency p50 / p95 | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules only | – | 100.0% | 2.5% | 4.9% | 0.0% | 12.7% | 0.0% | <1 ms | – |
| SafeGate (DeBERTa-v3-small, int8) | 0.975 | 97.3% | 90.0% | 93.5% | 0.3% | 84.7% | 4.7% | 54.55 / 314.44 ms | 176 MB |
| deberta-v3-base-prompt-injection-v2 (as published, PyTorch fp32) | 0.500 | 95.1% | 72.5% | 82.3% | 0.5% | 100.0% | 6.7% | 167.56 / 934.77 ms | – |
| deberta-v3-base-prompt-injection-v2 (fp32, threshold tuned on our validation split) | 0.021 | 93.8% | 75.0% | 83.3% | 0.7% | 100.0% | 8.0% | 169.11 / 930.1 ms | – |
| deberta-v3-base-prompt-injection-v2 (ONNX int8, like SafeGate) | 0.500 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 107.85 / 618.18 ms | 248 MB |

Training datasets named in the baseline's model card that we also use:
- jackhhao/jailbreak-classification
