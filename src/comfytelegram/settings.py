"""Runtime configuration, loaded from environment variables / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

#: Durable local state (sqlite files) lives under one gitignored directory
#: rather than loose at the project root. Not just tidiness: Docker bind-
#: mounts a *missing host file* as an empty directory instead of creating a
#: file, which breaks `sqlite3.connect()` on first run — a directory-level
#: mount (`docker-compose.yml`'s `./data:/app/data`) doesn't have that
#: problem, since a missing host *directory* is exactly what Docker does
#: create automatically. Each `Storage`/`TagDatabase.__init__` still
#: `mkdir(parents=True, exist_ok=True)`s its own file's parent, so this
#: works identically for a bare `uv run comfytelegram` too.
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Bot-wide runtime config, populated from environment variables / `.env`
    (see `env.example` for the template). Field names map to env vars
    case-insensitively, e.g. `TELEGRAM_BOT_TOKEN` -> `telegram_bot_token`."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    telegram_bot_token: str = Field(..., description="Token from @BotFather")

    comfyui_host: str = Field("127.0.0.1", description="Host running ComfyUI")
    comfyui_port: int = Field(8188, description="ComfyUI HTTP/WS port")
    comfyui_use_tls: bool = Field(False, description="Use https/wss instead of http/ws")

    ollama_host: str = Field(
        "127.0.0.1", description="Host running Ollama (for Qwen-VL image analysis)"
    )
    ollama_port: int = Field(11434, description="Ollama HTTP port")
    ollama_vision_model: str = Field(
        "qwen3.5:4b",
        description=(
            "Ollama model tag to use for natural-language image captioning. Kept small "
            "by default since it has to coexist in VRAM with whatever checkpoint ComfyUI "
            "is keeping resident — bump to a bigger vision model (e.g. qwen3.6:27b) only "
            "if the GPU running this has VRAM to spare."
        ),
    )
    ollama_deep_vision_model: str = Field(
        "qwen3.8:latest",
        description=(
            "Ollama model tag for deep image analysis — a bigger, slower vision model "
            "than ollama_vision_model, for when the quick caption isn't detailed enough. "
            "Used unconditionally for directly-uploaded photos (not on any interactive "
            'generation path) and opt-in via the "🔎 Deep Analyze" button on the bot\'s '
            "own generated images. Not part of generation's quick-model dispatch, so "
            "it's fine for this to be too large to keep resident alongside a checkpoint."
        ),
    )

    wd14_model_repo: str = Field(
        "SmilingWolf/wd-vit-tagger-v3",
        description="Hugging Face repo providing the WD14 tagger's model.onnx + selected_tags.csv",
    )
    wd14_model_dir: Path = Field(
        default=PROJECT_ROOT / "models" / "wd14",
        description="Local cache directory for the downloaded WD14 tagger files",
    )
    wd14_tag_threshold: float = Field(
        0.35, description="Minimum WD14 tag confidence to include in a derived prompt"
    )

    model_profiles_dir: Path = Field(
        default=PROJECT_ROOT / "model_profiles",
        description="Directory of per-checkpoint default-settings JSON files",
    )
    state_db_path: Path = Field(
        default=DATA_DIR / "state.sqlite3",
        description="SQLite file for durable per-chat state (selected model, profile overrides)",
    )

    tags_db_path: Path = Field(
        default=DATA_DIR / "tags.sqlite3",
        description=(
            "SQLite file for the local danbooru/e621 tag database backing /tags and "
            "/tagcheck. Separate from state_db_path since it's bulk reference data "
            "wholesale-replaced by an import, not per-chat state. Self-populates on "
            "first startup unless tag_db_auto_update is disabled — see that field."
        ),
    )
    tag_db_auto_update: bool = Field(
        True,
        description=(
            "Automatically refresh any tag source (danbooru/e621) that's missing or "
            "older than tag_db_max_age_days, in the background at bot startup (doesn't "
            "block startup; a failed fetch — e.g. no network — just logs a warning and "
            "leaves whatever was already imported). Disable for an offline/airgapped "
            "install and populate tags_db_path with scripts/update_tag_db.py instead."
        ),
    )
    tag_db_max_age_days: float = Field(
        30,
        description=(
            "How old a tag source's last import can get before the startup auto-update "
            "refreshes it — matches the upstream archive's own monthly refresh cadence."
        ),
    )
    tag_rare_threshold: int = Field(
        100,
        description="post_count floor below which /tagcheck flags a tag as rare (little training data)",
    )
    tag_search_results: int = Field(15, description="Max rows /tags replies with per query")

    inpaint_relay_url: str | None = Field(
        None,
        description=(
            "Base URL of the public inpaint_relay deployment (see inpaint_relay/ at "
            'the repo root) — e.g. https://inpaint.example.com. Backs the "🖌️ Draw '
            'Mask" hand-detail option, which opens a Telegram WebApp there for '
            "freehand mask drawing. The button is omitted entirely when this is unset, "
            "since the relay has to be reachable over the public internet for "
            "Telegram's client to open it — there's no local fallback."
        ),
    )
    inpaint_relay_shared_secret: str | None = Field(
        None,
        description=(
            "Bearer secret shared with inpaint_relay's own INPAINT_RELAY_SHARED_SECRET "
            "env var, authenticating comfytelegram's job-management calls (creating a "
            "job, polling/pulling its result, deleting it). Never sent to the browser — "
            "the relay's own webapp page and image/mask endpoints are reached by "
            "unguessable per-job token instead, the same trust model as a Telegram "
            "file_id download link."
        ),
    )
    inpaint_poll_interval_seconds: float = Field(
        3.0,
        description=(
            "How often the background poller checks inpaint_relay for a finished mask "
            "drawing, per pending job (see storage.py's inpaint_job table). The relay "
            "can't push to comfytelegram directly — the home box isn't reachable from "
            "the public internet — so this is outbound polling, the same posture as "
            "Telegram's own run_polling()."
        ),
    )

    @field_validator("inpaint_relay_url")
    @classmethod
    def _require_scheme_and_strip_trailing_slash(cls, value: str | None) -> str | None:
        """A bare hostname (e.g. `inpaint.example.com`, missing `https://`)
        passes `str` validation fine but breaks every outbound relay call
        with an opaque `aiohttp.InvalidUrlClientError` at the first button
        tap/poll tick instead of a clear error at startup — fail loudly
        here instead. Also strips one trailing slash so
        `f"{url}/jobs"`-style building elsewhere never produces `//jobs`."""
        if value is None:
            return value
        if not value.startswith(("http://", "https://")):
            raise ValueError(
                f"INPAINT_RELAY_URL must start with http:// or https:// (got {value!r})"
            )
        return value.rstrip("/")

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
        """Parse the env var's `"123,456"` form into `[123, 456]`; empty
        segments are dropped, so a trailing comma or blank string is
        harmless. Non-string values (e.g. already a list) pass through
        unchanged."""
        if isinstance(value, str):
            return [int(part) for part in value.split(",") if part.strip()]
        return value

    @property
    def comfyui_http_base(self) -> str:
        """HTTP base URL for the configured ComfyUI instance, e.g.
        `http://127.0.0.1:8188`."""
        scheme = "https" if self.comfyui_use_tls else "http"
        return f"{scheme}://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def comfyui_ws_base(self) -> str:
        """WebSocket base URL for the configured ComfyUI instance, e.g.
        `ws://127.0.0.1:8188`."""
        scheme = "wss" if self.comfyui_use_tls else "ws"
        return f"{scheme}://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def ollama_http_base(self) -> str:
        """HTTP base URL for the configured Ollama instance, e.g.
        `http://127.0.0.1:11434`."""
        return f"http://{self.ollama_host}:{self.ollama_port}"


def load_settings() -> Settings:
    """Construct `Settings` from environment variables / `.env`."""
    return Settings()
