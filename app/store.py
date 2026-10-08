import hashlib
from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.pipeline import PipelineResult, overall_action


class Base(DeclarativeBase):
    pass


class Decision(Base):
    """One row per screened request. Raw prompts are never stored, only a SHA-256 of the input."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    app: Mapped[str] = mapped_column(String(64))
    endpoint: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(16))
    blocked_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    input_sha256: Mapped[str] = mapped_column(String(64))
    screen_ms: Mapped[float] = mapped_column(Float)
    verdicts: Mapped[list] = mapped_column(JSON)

    __table_args__ = (
        Index("ix_decisions_created_at", "created_at"),
        Index("ix_decisions_app_action", "app", "action"),
    )

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            "app": self.app,
            "endpoint": self.endpoint,
            "action": self.action,
            "blocked_by": self.blocked_by,
            "input_sha256": self.input_sha256,
            "screen_ms": self.screen_ms,
            "verdicts": self.verdicts,
        }


class DecisionStore:
    def __init__(self, database_url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(database_url)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self) -> None:
        # Week 1 uses create_all; switch to Alembic migrations once the schema settles.
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def record(
        self,
        *,
        request_id: str,
        app: str,
        endpoint: str,
        input_text: str,
        results: list[PipelineResult],
        screen_ms: float,
    ) -> None:
        verdicts = [
            {**{k: v for k, v in asdict(verdict).items() if k != "redacted_text"}, "stage": r.stage}
            for r in results
            for verdict in r.verdicts
        ]
        blocker = next((r for r in results if r.blocked_by), None)
        row = Decision(
            request_id=request_id,
            created_at=datetime.now(UTC),
            app=app,
            endpoint=endpoint,
            action=overall_action(results).value,
            blocked_by=_rail_label(blocker.blocked_by.rail, blocker.stage) if blocker else None,
            input_sha256=hashlib.sha256(input_text.encode()).hexdigest(),
            screen_ms=round(screen_ms, 3),
            verdicts=verdicts,
        )
        async with self.sessions() as session:
            session.add(row)
            await session.commit()

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        app: str | None = None,
        action: str | None = None,
        rail: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict]:
        query = select(Decision).order_by(Decision.id.desc()).limit(limit).offset(offset)
        if app:
            query = query.where(Decision.app == app)
        if action:
            query = query.where(Decision.action == action)
        if rail:
            query = query.where(Decision.blocked_by == rail)
        if since:
            query = query.where(Decision.created_at >= since)
        if until:
            query = query.where(Decision.created_at < until)
        async with self.sessions() as session:
            rows = (await session.scalars(query)).all()
        return [row.to_dict() for row in rows]

    async def stats(self, window: int = 1000) -> dict:
        """Summary of the most recent `window` decisions for the dashboard."""
        query = (
            select(Decision.action, Decision.blocked_by, Decision.screen_ms, Decision.verdicts)
            .order_by(Decision.id.desc())
            .limit(window)
        )
        async with self.sessions() as session:
            total = await session.scalar(select(func.count(Decision.id)))
            rows = (await session.execute(query)).all()
        by_action = {"allow": 0, "redact": 0, "block": 0}
        blocks_by_rail: dict[str, int] = {}
        redactions_by_rail: dict[str, int] = {}
        latencies = sorted(r.screen_ms for r in rows)
        for row in rows:
            by_action[row.action] = by_action.get(row.action, 0) + 1
            if row.blocked_by:
                blocks_by_rail[row.blocked_by] = blocks_by_rail.get(row.blocked_by, 0) + 1
            for verdict in row.verdicts:
                if verdict.get("action") == "redact":
                    rail = _rail_label(verdict.get("rail", "?"), verdict.get("stage", "input"))
                    redactions_by_rail[rail] = redactions_by_rail.get(rail, 0) + 1
        return {
            "total": total or 0,
            "window": len(rows),
            "by_action": by_action,
            "blocks_by_rail": blocks_by_rail,
            "redactions_by_rail": redactions_by_rail,
            "screen_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
        }


def _rail_label(rail: str, stage: str) -> str:
    """Output rails are labelled "output:<rail>" so they don't merge with input rails."""
    return rail if stage == "input" else f"{stage}:{rail}"


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1)))
    return round(sorted_values[index], 3)
