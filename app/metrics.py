from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "safegate_requests_total", "Requests handled by SafeGate", ["endpoint", "action"]
)
RAIL_ACTIONS = Counter(
    "safegate_rail_actions_total", "Verdicts returned by each rail", ["rail", "stage", "action"]
)
RAIL_LATENCY = Histogram(
    "safegate_rail_seconds",
    "Time spent in each rail",
    ["rail", "stage"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
SCREEN_LATENCY = Histogram(
    "safegate_screen_seconds",
    "Total time spent screening one request (input and output rails)",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
