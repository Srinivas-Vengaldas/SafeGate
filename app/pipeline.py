import time
from dataclasses import dataclass, field

from app import metrics
from app.rails.base import Action, Rail, Verdict


@dataclass
class PipelineResult:
    text: str
    verdicts: list[Verdict] = field(default_factory=list)
    stage: str = "input"  # input (the prompt) or output (the model's reply)

    @property
    def blocked_by(self) -> Verdict | None:
        return next((v for v in self.verdicts if v.action is Action.BLOCK), None)

    @property
    def action(self) -> Action:
        if self.blocked_by:
            return Action.BLOCK
        if any(v.action is Action.REDACT for v in self.verdicts):
            return Action.REDACT
        return Action.ALLOW


def run_rails(rails: list[Rail], text: str, stage: str = "input") -> PipelineResult:
    """Run rails in order. A block stops the chain; a redact feeds its text to later rails."""
    result = PipelineResult(text=text, stage=stage)
    for rail in rails:
        started = time.perf_counter()
        verdict = rail.check(result.text)
        metrics.RAIL_LATENCY.labels(rail.name, stage).observe(time.perf_counter() - started)
        metrics.RAIL_ACTIONS.labels(rail.name, stage, verdict.action.value).inc()
        result.verdicts.append(verdict)
        if verdict.action is Action.BLOCK:
            break
        if verdict.action is Action.REDACT and verdict.redacted_text is not None:
            result.text = verdict.redacted_text
    return result


def overall_action(results: list[PipelineResult]) -> Action:
    """Combine per-text results: any block wins, then any redact, else allow."""
    actions = {r.action for r in results}
    for action in (Action.BLOCK, Action.REDACT):
        if action in actions:
            return action
    return Action.ALLOW
