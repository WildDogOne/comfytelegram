"""Public-host relay backing comfytelegram's "🖌️ Draw Mask" hand-detail
option (see the main repo's `handlers.py`/`Settings.inpaint_relay_url`).

Deliberately dumb and generic: this service never talks to ComfyUI, never
sees the Telegram bot token, and doesn't know what a "hand" or a "mask" is
for — it just shuttles two blobs (a source image in, a drawn mask out)
between comfytelegram (reachable only outbound, from a home box with no
public exposure) and a Telegram client opening this page as a WebApp
(https://core.telegram.org/bots/webapps).

Job state lives in memory only — a restart here just orphans whatever jobs
were in flight; comfytelegram's own `inpaint_job` table times them out on
its side instead of waiting forever. See the main repo's plan/README for
the full architecture.

Two trust boundaries:
  - `POST /jobs`, `GET /jobs/{token}/result`, `DELETE /jobs/{token}` are
    gated by a shared secret (`INPAINT_RELAY_SHARED_SECRET`) known only to
    comfytelegram and this relay.
  - `GET /jobs/{token}`, `GET /jobs/{token}/image`, `POST /jobs/{token}/mask`
    have no such auth — they're the URLs Telegram's client itself opens/
    calls from the user's phone, so they're only as secret as the
    unguessable per-job token (same trust model as a Telegram file_id
    download link). This relay can't validate that a submitted mask
    actually came from Telegram (it doesn't hold the bot token) — that
    check happens back on comfytelegram, against `Telegram.WebApp.initData`,
    before anything here is acted on.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: How long a job can sit undrawn (or drawn-but-unpicked-up) before it's
#: dropped from memory — an abandoned WebApp tab, or comfytelegram being
#: down, shouldn't grow this dict forever. Doesn't need to match
#: comfytelegram's own `INPAINT_JOB_TTL_SECONDS` — whichever side times a
#: job out first just means the other one gets a 404/None on its next
#: check, and both already handle that.
JOB_TTL_SECONDS = float(os.environ.get("INPAINT_RELAY_JOB_TTL_SECONDS", "1800"))


@dataclass
class Job:
    #: Re-encoded (see `_to_display_jpeg`) purely for faster delivery to the
    #: mask editor's <canvas> — never the thing anything gets generated
    #: from. `image_content_type` names what it actually got encoded as,
    #: since re-encoding falls back to serving the original bytes verbatim
    #: (still PNG) if Pillow can't decode them for whatever reason.
    image: bytes
    image_content_type: str
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # "pending" -> "submitted"
    mask: bytes | None = None
    init_data: str | None = None


def _to_display_jpeg(source: bytes) -> bytes:
    """Re-encode `source` as a quality-85 JPEG at its *original* pixel
    dimensions — dropping only what JPEG-vs-PNG compression itself drops,
    never resizing — purely so `GET /jobs/{token}/image` has far fewer
    bytes to ship to the phone opening the mask editor. Safe to do
    unconditionally: this copy is only ever drawn onto the editor's
    on-screen `<canvas>` for visual reference while the user paints: the
    mask they draw is submitted as its own separate grayscale PNG
    (`POST /jobs/{token}/mask`), sized off `image.naturalWidth/Height` in
    the browser (unchanged by re-encoding, since dimensions aren't
    touched), and comfytelegram composites that mask against *its own*
    original full-quality PNG upload, never this relay's copy — so nothing
    downstream ever sees this image's compression artifacts."""
    with Image.open(io.BytesIO(source)) as im:
        rgb = im.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


app = FastAPI(title="inpaint-relay")
_jobs: dict[str, Job] = {}


def _shared_secret() -> str:
    """Read the shared secret fresh from the environment on every check
    rather than caching it at import time, so a `docker compose up` that
    starts this before the env var is actually populated (or a bad
    deployment that forgets to set it) fails every authenticated request
    loudly instead of silently running with `None == None`-style auth."""
    secret = os.environ.get("INPAINT_RELAY_SHARED_SECRET")
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="INPAINT_RELAY_SHARED_SECRET is not configured on this relay",
        )
    return secret


def require_shared_secret(request: Request) -> None:
    """FastAPI dependency gating comfytelegram's own job-management calls
    (create/poll/delete) — never applied to the routes Telegram's client
    itself opens/calls (see module docstring)."""
    expected = _shared_secret()
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _prune_expired_jobs() -> None:
    """Drop jobs older than `JOB_TTL_SECONDS`. Called at the top of every
    request rather than on a separate background timer — this dict is
    small (one entry per in-flight mask draw across the whole bot), so an
    O(n) scan per request is cheap, and it keeps this file free of any
    asyncio-task-lifecycle bookkeeping."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for token in [token for token, job in _jobs.items() if job.created_at < cutoff]:
        del _jobs[token]


@app.post("/jobs", dependencies=[Depends(require_shared_secret)])
async def create_job(request: Request) -> dict[str, str]:
    """comfytelegram pushes the source image here (raw bytes body) and gets
    back an unguessable token to build the WebApp button's URL from."""
    _prune_expired_jobs()
    image = await request.body()
    if not image:
        raise HTTPException(status_code=400, detail="Empty request body")
    try:
        display_image = _to_display_jpeg(image)
        content_type = "image/jpeg"
    except Exception:
        logger.warning("Couldn't re-encode uploaded image as JPEG; serving it as-is", exc_info=True)
        display_image = image
        content_type = "image/png"
    token = secrets.token_urlsafe(24)
    _jobs[token] = Job(image=display_image, image_content_type=content_type)
    return {"token": token}


@app.get("/jobs/{token}")
async def mask_editor_page(token: str) -> FileResponse:
    """The page Telegram opens as a WebApp — a static file, not templated;
    its own JS reads `token` back out of `location.pathname`. 404s (rather
    than silently serving the editor) once the job has expired or been
    consumed, since there'd be nothing left for it to load."""
    _prune_expired_jobs()
    if token not in _jobs:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/jobs/{token}/image")
async def job_image(token: str) -> Response:
    """The source image for the mask editor's `<canvas>` to draw over."""
    _prune_expired_jobs()
    job = _jobs.get(token)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    return Response(content=job.image, media_type=job.image_content_type)


@app.post("/jobs/{token}/mask")
async def submit_mask(token: str, request: Request) -> dict[str, bool]:
    """The mask editor's "Done" button POSTs here: a multipart body with a
    `mask` file field (grayscale PNG, white = inpaint) and an `init_data`
    text field (`Telegram.WebApp.initData`, opaque to this relay — see
    module docstring for why it's forwarded rather than checked here)."""
    _prune_expired_jobs()
    job = _jobs.get(token)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    form = await request.form()
    mask_file = form.get("mask")
    init_data = form.get("init_data")
    if mask_file is None or init_data is None:
        raise HTTPException(status_code=400, detail="Both 'mask' and 'init_data' are required")
    job.mask = await mask_file.read()
    job.init_data = str(init_data)
    job.status = "submitted"
    return {"ok": True}


@app.get("/jobs/{token}/result", dependencies=[Depends(require_shared_secret)])
async def job_result(token: str) -> JSONResponse:
    """comfytelegram polls this to find out whether a mask has been drawn
    yet. Base64-encodes the mask so the whole response can stay a single
    JSON body alongside `init_data`, rather than needing a second endpoint
    or a multipart response just for the bytes."""
    _prune_expired_jobs()
    job = _jobs.get(token)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    if job.status != "submitted":
        return JSONResponse({"status": "pending"})
    return JSONResponse(
        {
            "status": "submitted",
            "mask_base64": base64.b64encode(job.mask).decode("ascii"),
            "init_data": job.init_data,
        }
    )


@app.delete("/jobs/{token}", dependencies=[Depends(require_shared_secret)])
async def delete_job(token: str) -> dict[str, bool]:
    """comfytelegram calls this once it's pulled a job's result (or given
    up on it) — best-effort from its side, so a missing token here is fine,
    not an error."""
    _jobs.pop(token, None)
    return {"ok": True}
