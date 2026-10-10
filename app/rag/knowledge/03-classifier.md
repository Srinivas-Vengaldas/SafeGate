# The injection classifier

## Model and training data
The injection rail is microsoft/deberta-v3-small fine-tuned for three epochs on CPU in GitHub Actions. Training data comes from public datasets: deepset/prompt-injections, jackhhao/jailbreak-classification, 2,500 benign instructions sampled from databricks-dolly-15k, and, from version 2 on, 2,500 benign coding instructions from CodeAlpaca-20k used as hard negatives. Lakera's gandalf_ignore_instructions dataset is fully held out from training to test how well the model generalizes to attacks from a source it never saw.

## Avoiding leakage
Exact and near-duplicate prompts are collapsed before the data is split, using MinHash with a Jaccard similarity of at least 0.8 on word 5-grams. Groups with conflicting labels are dropped. Held-out prompts that nearly duplicate anything on the training side are removed, and training prompts that nearly match the hand-written tricky-benign evaluation set are removed too. The blocking threshold is chosen on the validation split, never on a test set.

## The false positive problem and version 2
The first version over-blocked safe prompts that use attack-like words. It flagged 28 of 150 hand-written safe prompts, such as asking to compare two strings while not caring about letter case, because its training data had almost no benign prompts with words like ignore, override or shell commands. Version 2 added 2,500 benign coding instructions as hard negatives and cut the tricky-benign false positive rate from 18.7% to 4.7%, at the cost of some recall. Version 3, the one served now, added about 3,600 Wikipedia passages, some with an instruction hidden in them and some with ordinary instructions for the human reader, so that it recognizes attacks inside retrieved documents.

## Serving on CPU with ONNX
The served model is exported to ONNX with int8 weights and runs on ONNX Runtime without PyTorch. Quantization cut CPU latency from 65 to 43 ms at the median and from 343 to 209 ms at p95, and shrank the weights from about 540 MB to 172 MB. Very long prompts are scored on at most four windows of 256 tokens, the first and the last. ONNX Runtime memory-maps the weights instead of copying them onto the heap, which keeps the whole demo container at about 390 MB, under the 512 MB free-tier limit.

## Choosing the threshold
The threshold trades recall for fewer false positives. For the PyTorch model, the validation-chosen threshold of 0.47 gives 87.2% recall on the held-out Gandalf attacks with 4.7% false positives on tricky-benign prompts; a threshold of 0.90 gives 83.8% and 2.0%, and 0.99 gives 78.2% and 0.7%. The int8 model's validation-chosen threshold is 0.975.
