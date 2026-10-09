# Files, images and the RAG pipeline

## Screening images and screenshots
Images attached to a chat request are read with Tesseract OCR. If the rails redact anything in the extracted text, such as an email address or an API key, SafeGate blacks out those words in the image itself before forwarding it. Every image is re-encoded without metadata, which removes GPS location, camera details and timestamps.

## Screening documents
PDFs (their text layer), Word files and plain-text files are read and screened in pieces of 4,000 characters, each like a prompt. A document with redactions is forwarded to the LLM as its redacted text. An injection hidden inside a document blocks the whole request before the LLM is called. Attachments SafeGate cannot read, such as remote image URLs, uploaded file ids, encrypted PDFs or unknown formats, are blocked by default, because SafeGate cannot vouch for what it cannot read. Scanned PDFs without a text layer are not OCR'd yet.

## The guarded RAG pipeline
SafeGate's retrieval-augmented generation pipeline puts the rails at every point where untrusted text can enter. An uploaded document is extracted, screened with the input rails, split into overlapping passages of about 700 characters, embedded and indexed; a document that fails screening is never indexed, and one with personal data is indexed redacted. A question is screened, embedded, and matched against the index. Each retrieved passage is screened again before it enters the prompt, and passages that fail are dropped. The LLM is told the sources are data rather than instructions and must cite them by number, citations of sources that do not exist are removed, and the answer passes through the output rails.

## Why passages are screened again
Text can reach a knowledge base without passing through upload screening: a synced wiki, a shared drive, or documents indexed before the policy changed. Screening at retrieval time is the defense against this kind of indirect injection, where an instruction hidden in a document tries to take over the model. The demo's sample knowledge base plants one such document, indexed without screening, so visitors can watch it being dropped before it reaches the prompt.

## Models and privacy in the RAG demo
Answers come from any OpenAI-compatible endpoint; the public demo uses Google's Gemini through its OpenAI-compatible API. Retrieval uses an embedding model when one is configured, and otherwise a built-in hashing embedder that matches shared words without any model. Each visitor's documents live in memory in their own collection, keyed by a random id kept in their browser, and expire after an hour without use. Generated answers are capped per visitor and per day, since a server-side API key pays for them.
