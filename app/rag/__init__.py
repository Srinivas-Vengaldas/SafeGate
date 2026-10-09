"""A small retrieval-augmented generation pipeline with SafeGate's rails at every step:
documents are screened before they are indexed, retrieved passages are screened again before
they reach the prompt, and the answer is screened before the user sees it."""
