## Poisoned retrieval

400 clean Wikipedia passages (Dolly contexts) and 260 poisoned ones (gandalf 150, addressed 80, unaddressed 30), each screened as a retrieved passage.

| Rails | Caught: all | gandalf | addressed | unaddressed | Clean wrongly dropped | p50 per passage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Document rails (default context rails) | 27.7% | 28.0% | 37.5% | 0.0% | 0.0% | 1.2 ms |
| Prompt rails (rules + classifier) | 43.8% | 54.7% | 40.0% | 0.0% | 0.5% | 47.5 ms |
| Classifier only | 42.7% | 53.3% | 38.8% | 0.0% | 0.5% | 45.6 ms |
| Document rails + classifier | 52.3% | 57.3% | 62.5% | 0.0% | 0.5% | 46.0 ms |
