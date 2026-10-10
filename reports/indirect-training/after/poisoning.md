## Poisoned retrieval

400 clean Wikipedia passages (Dolly contexts) and 260 poisoned ones (gandalf 150, addressed 80, unaddressed 30), each screened as a retrieved passage, plus the 39 passages of the demo's own knowledge bases.

| Rails | Caught: all | gandalf | addressed | unaddressed | Clean wrongly dropped | Demo docs wrongly dropped | p50 per passage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Document rails (default context rails) | 27.7% | 28.0% | 37.5% | 0.0% | 0.0% | 0.0% | 2.1 ms |
| Prompt rails (rules + classifier) | 96.5% | 95.3% | 100.0% | 93.3% | 1.5% | 41.0% | 92.3 ms |
| Classifier only | 96.5% | 95.3% | 100.0% | 93.3% | 0.8% | 41.0% | 90.2 ms |
| Document rails + classifier | 96.5% | 95.3% | 100.0% | 93.3% | 1.5% | 41.0% | 90.3 ms |
