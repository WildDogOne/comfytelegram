"""Runtime configuration, loaded from environment variables / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    telegram_bot_token: str = Field(..., description="Token from @BotFather")

    comfyui_host: str = Field("127.0.0.1", description="Host running ComfyUI")
    comfyui_port: int = Field(8188, description="ComfyUI HTTP/WS port")
    comfyui_use_tls: bool = Field(False, description="Use https/wss instead of http/ws")

    model_profiles_dir: Path = Field(
        default=PROJECT_ROOT / "model_profiles",
        description="Directory of per-checkpoint default-settings JSON files",
    )
    state_db_path: Path = Field(
        default=PROJECT_ROOT / "state.sqlite3",
        description="SQLite file for durable per-chat state (selected model, profile overrides)",
    )

    # NoDecode: pydantic-settings would otherwise try to JSON-decode this env
    # var before validation ever sees it (its default behavior for list-typed
    # fields), rejecting the plain "123,456" form env.example documents.
    allowed_user_ids: Annotated[list[int], NoDecode] = Field(
        default_factory=list,
        description="If non-empty, only these Telegram user IDs may use the bot",
    )

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def _parse_comma_separated_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return [int(part) for part in value.split(",") if part.strip()]
        return value

    @property
    def comfyui_http_base(self) -> str:
        scheme = "https" if self.comfyui_use_tls else "http"
        return f"{scheme}://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def comfyui_ws_base(self) -> str:
        scheme = "wss" if self.comfyui_use_tls else "ws"
        return f"{scheme}://{self.comfyui_host}:{self.comfyui_port}"


def load_settings() -> Settings:
    return Settings()
