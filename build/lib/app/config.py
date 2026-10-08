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


@lru_cache
def get_settings() -> Settings:
    return Settings()
