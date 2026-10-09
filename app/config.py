from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAFEGATE_", env_file=".env", extra="ignore")

    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str = ""
    upstream_timeout_s: float = 60.0
    database_url: str = "sqlite+aiosqlite:///./safegate.db"
    policy_dir: str = "policies"
    spacy_model: str = "en_core_web_sm"
    # Shared rate-limit counters and verdict cache. Empty keeps both in process memory.
    redis_url: str = ""
    cache_ttl_s: int = 3600  # 0 disables the verdict cache
    cache_max_entries: int = 10_000  # in-memory cache only
    # Rate-limit clients by the first X-Forwarded-For address. Enable only behind a proxy that
    # sets it (Render, a load balancer); otherwise clients could pick their own identity.
    trust_forwarded_for: bool = False

    # Retrieval-augmented answers (/v1/rag/*). Any OpenAI-compatible endpoint works; for Gemini
    # use https://generativelanguage.googleapis.com/v1beta/openai and a Gemini model name.
    # Empty base URL and key fall back to the upstream ones. Without any key, queries return the
    # screened passages but no generated answer.
    rag_base_url: str = ""
    rag_api_key: str = ""
    # Comma-separated: later models answer when earlier ones are busy (rate limited, overloaded).
    rag_chat_model: str = "gpt-4o-mini"
    # Passed as reasoning_effort to thinking models (e.g. "low"); empty sends nothing.
    rag_reasoning_effort: str = ""
    # Empty: a built-in hashing embedder (matches shared words, needs no model or key).
    rag_embed_model: str = ""
    rag_top_k: int = 4
    rag_ttl_s: int = 3600  # collections are dropped after this long without use
    rag_max_documents: int = 20  # per collection
    rag_max_passages: int = 300  # per collection
    rag_max_collections: int = 50
    # Generated answers per client per minute, and across all clients per day: a server-side
    # key pays for every answer, so these cap what a public demo can spend.
    rag_answers_per_minute: int = 6
    rag_answers_per_day: int = 500

    @property
    def rag_url(self) -> str:
        return self.rag_base_url or self.upstream_base_url

    @property
    def rag_chat_models(self) -> list[str]:
        return [m.strip() for m in self.rag_chat_model.split(",") if m.strip()]

    @property
    def rag_key(self) -> str:
        return self.rag_api_key or self.upstream_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
