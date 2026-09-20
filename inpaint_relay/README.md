# inpaint_relay

Small public-host relay backing comfytelegram's "🖌️ Draw Mask" hand-detail
option. It's what makes a [Telegram Web App](https://core.telegram.org/bots/webapps)
mask editor possible even though comfytelegram itself runs on a box with no
public/inbound network exposure — see `server.py`'s module docstring for the
full trust model and the main repo's `CLAUDE.md`/plan for the end-to-end
architecture.

This is a separate, independently-deployed service: its own `pyproject.toml`,
its own `Dockerfile`, meant to run on your public Traefik host, not alongside
the bot itself.

## What it does (and doesn't)

- Holds two blobs per in-flight job, in memory only: the source image
  (pushed by comfytelegram) and the drawn mask (pushed by the browser).
- Serves the mask-editor page (`static/index.html`) and the source image to
  whatever opens the per-job URL — no ComfyUI or Telegram bot-token access
  needed or wanted here.
- Does **not** validate that a submitted mask genuinely came from Telegram
  (`Telegram.WebApp.initData`) — it has no way to, since it never holds the
  bot token. comfytelegram validates that itself after pulling a job back
  (see `auth.validate_webapp_init_data` in the main package). Don't skip
  that check when integrating this with anything else.
- Does not persist anything to disk — a restart drops every in-flight job.
  That's fine for its purpose (a job only needs to live for as long as
  someone's actively drawing).

## Running it

```bash
cd inpaint_relay
uv sync
cp env.example .env              # then edit .env — see below
uv run --env-file .env uvicorn server:app --reload
```

Or via Docker (which reads `.env` through `docker-compose.example.yml`'s
`env_file:`, or pass `-e` flags directly):

```bash
docker build -t inpaint-relay .
docker run -p 8000:8000 --env-file .env inpaint-relay
```

`.env` (copied from `env.example`) needs:

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `INPAINT_RELAY_SHARED_SECRET` | yes | — | must match `INPAINT_RELAY_SHARED_SECRET` in comfytelegram's own `.env` — authenticates its job-management calls (`POST /jobs`, `GET /jobs/{token}/result`, `DELETE /jobs/{token}`) |
| `INPAINT_RELAY_JOB_TTL_SECONDS` | no | `1800` | how long an undrawn/unpicked-up job lingers in memory before being dropped |
| `SUBDOMAIN` / `DOMAIN_NAME` | only for `docker-compose.example.yml` | — | used solely by its Traefik `Host(...)` label (see below) — not read by `server.py` itself |

## Deploying behind Traefik

See `docker-compose.example.yml` in this directory for the Traefik router
labels — same pattern as any other service on an existing Traefik setup
(`traefik.enable=true`, a `Host(...)` rule, `tls=true`, your certresolver).
Merge its `inpaint-relay` service block into your public host's actual
compose file (or run it as its own `docker compose -f
inpaint_relay/docker-compose.example.yml up -d` alongside it, on the same
Docker network Traefik watches) — whichever fits your existing setup. Point
comfytelegram's `INPAINT_RELAY_URL` at the resulting public URL, e.g.
`https://inpaint.example.com`.

No TLS termination happens in this service itself — it expects Traefik (or
whatever's in front of it) to handle that, and just serves plain HTTP on
port 8000.
