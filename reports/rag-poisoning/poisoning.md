## Poisoned retrieval

400 clean Wikipedia passages (Dolly contexts) and 260 poisoned ones (gandalf 150, addressed 80, unaddressed 30), each screened as a retrieved passage, plus the 39 passages of the demo's own knowledge bases.

| Rails | Caught: all | gandalf | addressed | unaddressed | Clean wrongly dropped | Demo docs wrongly dropped | p50 per passage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Document rails (default context rails) | 27.7% | 28.0% | 37.5% | 0.0% | 0.0% | 0.0% | 2.1 ms |
| Prompt rails (rules + classifier) | 44.6% | 56.0% | 40.0% | 0.0% | 0.8% | 59.0% | 90.0 ms |
| Classifier only | 43.5% | 54.7% | 38.8% | 0.0% | 0.8% | 59.0% | 88.6 ms |
| Document rails + classifier | 53.8% | 58.7% | 65.0% | 0.0% | 0.8% | 59.0% | 88.2 ms |
