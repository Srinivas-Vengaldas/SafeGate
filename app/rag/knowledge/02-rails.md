# How the rails work

## Rails and verdicts
A rail is one check. Each rail returns a verdict of allow, redact or block, with a score and a human-readable reason. Rails run in the order the policy lists them, cheapest first, so most attacks are stopped by fast checks before the expensive classifier runs. A block stops the chain and the LLM is never called. A redaction passes the cleaned text to every later rail and then to the LLM, so the model only ever sees placeholders such as <EMAIL_ADDRESS> instead of the original value.

## Input rails
The rules rail enforces a maximum prompt length, a denylist of known attack phrases and a set of regular expressions. It runs in under a millisecond but only catches exact phrasings. The secrets rail recognizes the documented formats of common API keys, access tokens and private keys, and also passwords written out in a sentence. The PII rail uses Microsoft Presidio to find email addresses, phone numbers, US social security numbers, credit card numbers, IBANs and IP addresses, and redacts them by default. The injection rail is a fine-tuned DeBERTa-v3-small classifier that catches reworded attacks the rules miss.

## Output rails
Output rails screen the model's reply before the client sees it. The default policy redacts secrets and personal data in replies, and the demo also runs a toxicity rail based on Detoxify's model, which withholds a toxic reply and returns a short explanation instead. When any output rail is set, streamed replies are buffered, screened and then replayed, so the time to first token becomes the full generation time. That is the cost of never showing a client unscreened text.

## Secrets in plain sentences
Besides key formats, the secrets rail catches made-up passwords written in prose, for example a wifi password mentioned in a sentence. It looks at words near credential terms such as password, passcode or token, and uses an English word list plus a character trigram language model to tell made-up strings apart from real words and technical jargon. A password that is an ordinary dictionary word still gets through, which is a known limitation.

## Policies
Policies are YAML files, one per application, selected with the X-SafeGate-App header. A policy lists the input and output rails with their settings, which message roles to screen, what to do on a block (return an error or a refusal), attachment limits and an optional rate limit per client. Unknown applications fall back to the default policy.

## Caching and rate limiting
Screening is deterministic, so verdicts are cached by policy, stage and text, and a repeated prompt skips the classifiers in about a millisecond. Rate limits count requests per client (by API key, otherwise by IP address) and per application. Both use Redis when it is configured, so several gateway replicas share counters and cache, and fall back to process memory otherwise. Redis errors fail open, because an outage of an auxiliary store should not take the gateway down.
