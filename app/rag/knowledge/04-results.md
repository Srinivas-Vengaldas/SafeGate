# Results and benchmarks

## Headline numbers
Behind SafeGate, the attack success rate of NVIDIA's garak red-teaming tool against a small LLM fell from 31.9% to 5.0%. The served injection classifier catches 84.7% of attacks from a dataset it never saw in training (Lakera Gandalf), wrongly flags 4.7% of hand-written safe prompts that look dangerous, and reaches a test F1 of 93.5%. The full rail stack adds about 45 ms at the median and 48 ms at p95 per request, and about 1 ms when the verdict is cached.

## Classifier evaluation
On the test split the current model has 91.2% recall, 97.3% precision and a 0.3% false positive rate. Its recall on the held-out Gandalf dataset is 87.2% for the PyTorch model and 83.8% to 84.7% for the served int8 model, which varies by about a point between CI runners with different CPUs. A rules-only baseline caught 2.5% of test attacks and 12.7% of held-out attacks, which shows that a denylist alone is not a defense.

## Comparison with a public model
SafeGate's classifier was compared with ProtectAI's deberta-v3-base-prompt-injection-v2, a widely used open model, on the same test sets and the same CPU. SafeGate scored a test F1 of 93.5% against 82.3%, test recall of 90.0% against 72.5%, and a tricky-benign false positive rate of 4.7% against 6.7%, at about a third of the latency (55 ms against 168 ms at the median). ProtectAI's model is better on the Gandalf dataset, with 100% recall against 84.7%. Caveats: ProtectAI's model card lists one of the datasets our test split draws from, and its int8 export scored every input as benign, so it was compared in its published fp32 form. The two models are complementary, and running both is a planned next step.

## Red-teaming with garak
Garak attacked qwen2.5:1.5b running on CPU, once directly and once behind SafeGate, with up to 40 prompts per probe. Prompt injection hijacks succeeded 75.0% of the time against the bare model and 3.3% behind SafeGate. Jailbreaks from the DanInTheWild set fell from 75.0% to 5.0%. Indirect injection hidden in documents fell only from 31.2% to 18.8%, which is SafeGate's weakest area and the reason for retrieval-time screening in the RAG pipeline. Toxic continuations fell from 4.0% to 1.5%. Overall, 153 of 480 attacks succeeded against the bare model and 24 of 480 behind SafeGate. This is one small model and one run, so the numbers are indicative.

## Poisoned retrieval
A benchmark hides one attack sentence in real Wikipedia passages and screens each passage as if it had been retrieved. SafeGate's default document rails caught 27.7% of 260 poisoned passages and wrongly dropped none of 400 clean ones, in about a millisecond each. Adding the injection classifier raised the catch rate to 52.3% while dropping 0.5% of clean passages. Attacks written as prompts were caught about half as often when hidden in a paragraph, and bare instructions with no addressee, such as recommending one product, got past every setup, which is the main open problem.

## Latency
The latency benchmark sends 1,000 chat completions one at a time to an instant mock LLM, so it measures only what SafeGate adds. The proxy alone adds 1.4 ms at the median. The full policy, with rules, PII and the injection classifier on the prompt and PII and toxicity on the reply, adds 44.8 ms at the median and 47.9 ms at p95. A cached verdict adds 1.3 ms. Two ONNX models in one process were first 3.7 times slower because ONNX Runtime's idle threads spun and stole each other's CPU cores; turning spinning off fixed it.
