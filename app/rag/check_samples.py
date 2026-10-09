"""Screens every passage of the sample knowledge bases with a policy's input rails and fails if
any passage is blocked other than in the planted documents, or a planted document gets through.

A retrained classifier or an edited document could otherwise make the demo drop a harmless
passage, or stop catching the planted one, without anyone noticing.

    python -m app.rag.check_samples [policy_dir]
"""

from __future__ import annotations

import sys

from app.pipeline import run_rails
from app.policy import PolicyRegistry
from app.rag.service import SAMPLE_SETS


def main() -> int:
    policy_dir = sys.argv[1] if len(sys.argv) > 1 else "policies"
    loaded = PolicyRegistry(policy_dir, "blank").get("default")
    failures = 0
    for name, make in SAMPLE_SETS.items():
        for document in make().documents:
            for passage in document.passages:
                result = run_rails(loaded.input_rails, passage, "context")
                blocked = result.blocked_by
                if bool(blocked) != document.planted:
                    failures += 1
                    reason = f"{blocked.rail}: {blocked.reason}" if blocked else "not blocked"
                    print(f"FAIL {name}/{document.name}: {reason}\n  {passage[:200]}")
                elif blocked:
                    print(f"ok   {name}/{document.name}: dropped by {blocked.rail}")
            if not document.planted:
                print(f"ok   {name}/{document.name}: {len(document.passages)} passages pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
