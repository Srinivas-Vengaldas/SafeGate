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


@lru_cache
def get_settings() -> Settings:
    return Settings()
