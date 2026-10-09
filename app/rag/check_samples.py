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
            for i, passage in enumerate(document.passages):
                result = run_rails(loaded.input_rails, passage, "context")
                blocked = result.blocked_by
                scores = " ".join(
                    f"{v.rail}={v.score:.3f}" for v in result.verdicts if v.rail == "injection"
                )
                ok = bool(blocked) == document.planted
                failures += not ok
                status = "ok  " if ok else "FAIL"
                verdict = f"blocked by {blocked.rail}" if blocked else "passes"
                print(f"{status} {name}/{document.name}#{i}: {verdict} {scores}".rstrip())
                if not ok:
                    print(f"     {passage[:160]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
