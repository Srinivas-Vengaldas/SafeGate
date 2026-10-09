garak 0.17.0 attack success rate against qwen2.5:1.5b on Ollama (lower is better), bare and
behind SafeGate with its full default policy (`on_block: refuse`). Up to 40 prompts per probe,
one generation each, temperature 0. Benchmark workflow run 37862709499; the raw garak reports
are in that run's `benchmark-reports` artifact.

| Attack category | Bare LLM | Behind SafeGate |
|---|---|---|
| Prompt injection (PromptInject hijacks) | 75.0% (90/120) | 3.3% (4/120) |
| Jailbreaks (DAN in the wild) | 75.0% (30/40) | 5.0% (2/40) |
| Encoded injection (Base64) | 0.0% (0/40) | 0.0% (0/40) |
| Indirect injection in documents | 31.2% (25/80) | 18.8% (15/80) |
| Toxic continuations (RealToxicityPrompts) | 4.0% (8/200) | 1.5% (3/200) |
| **All** | **31.9% (153/480)** | **5.0% (24/480)** |
