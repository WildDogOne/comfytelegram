FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install deps first so this layer is cached across source-only changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY model_profiles ./model_profiles
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "comfytelegram"]
