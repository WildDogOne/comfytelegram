"""Public-host relay backing comfytelegram's "🖌️ Draw Mask"/"🩹 Fix Artifact"
and "✏️ Detail Prompt" options (see the main repo's `handlers.py`/
`Settings.inpaint_relay_url`).

Deliberately dumb and generic: this service never talks to ComfyUI, never
sees the Telegram bot token, and doesn't know what a "hand" or a "mask" is
for — it just shuttles a source image in and a drawn mask (plus, for
"✏️ Detail Prompt", a one-shot prompt/denoise submitted alongside it) back
out, between comfytelegram (reachable only outbound, from a home box with
no public exposure) and a Telegram client opening this page as a WebApp
(https://core.telegram.org/bots/webapps). `Job.mode` ("mask" or
"mask_prompt") just picks whether the editor page also shows the
prompt/denoise fields next to its canvas — every job submits a mask either
way, through the same endpoint, and this relay never interprets any of it
beyond that one rendering decision.

Job state lives in memory only — a restart here just orphans whatever jobs
were in flight; comfytelegram's own `inpaint_job` table times them out on
its side instead of waiting forever. See the main repo's plan/README for
the full architecture.

Two trust boundaries:
  - `POST /jobs`, `GET /jobs/{token}/result`, `DELETE /jobs/{token}` are
    gated by a shared secret (`INPAINT_RELAY_SHARED_SECRET`) known only to
    comfytelegram and this relay.
  - `GET /jobs/{token}`, `GET /jobs/{token}/image`, `GET /jobs/{token}/meta`,
    `POST /jobs/{token}/mask` have no such auth — they're the URLs
    Telegram's client itself opens/calls from the user's phone, so they're
    only as secret as the unguessable per-job token (same trust model as a
    Telegram file_id download link). This relay can't validate that a
    submission actually came from Telegram (it doesn't hold the bot token)
    — that check happens back on comfytelegram, against
    `Telegram.WebApp.initData`, before anything here is acted on.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

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
    #: Exactly the bytes comfytelegram uploaded, served back verbatim to
    #: the editor page — never the thing anything gets generated from.
    #: Compression happens on the *bot* side before upload (see its
    #: `handlers._to_display_jpeg`), not here: this relay used to re-encode
    #: on arrival, which meant a ~19MB PNG crossed the internet only to be
    #: discarded at this end. `image_content_type` is sniffed from the
    #: bytes rather than trusted from the request, since it's what the
    #: browser is told to decode them as.
    image: bytes
    image_content_type: str
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # "pending" -> "submitted"
    #: "mask" (default, for "🖌️ Draw Mask"/"🩹 Fix Artifact" — just a canvas)
    #: or "mask_prompt" (for "✏️ Detail Prompt" — the same canvas, plus
    #: editable positive/negative/denoise fields shown alongside it). Set at
    #: creation from the `meta` comfytelegram sends (see `create_job`); pure
    #: rendering hint for the page, since both modes submit through the same
    #: `POST /jobs/{token}/mask`.
    mode: str = "mask"
    mask: bytes | None = None
    #: `positive_prompt`/`negative_prompt` start out holding whatever
    #: comfytelegram says this image's current prompt is (`create_job`'s
    #: `meta` — `GET /jobs/{token}/meta` hands them back so the editor can
    #: pre-fill its fields with something to edit down, rather than a blank
    #: box), for a `mode == "mask_prompt"` job only. `submit_mask`
    #: overwrites all three with whatever was actually submitted alongside
    #: the mask — `denoise` has no pre-fill (there's no "current" value to
    #: start it from, each detailer has its own fixed default), so it's
    #: `None` until then. One-shot either way: none of the three are saved
    #: anywhere beyond that one submission — see `DETAIL_PROMPT_CALLBACK_KIND`
    #: in the main package.
    positive_prompt: str | None = None
    negative_prompt: str | None = None
    denoise: float | None = None
    #: Read-only reference info for the page to display next to the
    #: editable fields (e.g. the detailer's own default steps/cfg/sampler/
    #: denoise) — set at creation from `meta`, this relay never reads it,
    #: just carries it through.
    readonly_info: dict[str, Any] | None = None
    init_data: str | None = None


#: Magic-byte prefixes for the image formats comfytelegram can upload,
#: mapped to what `GET /jobs/{token}/image` should claim they are.
_IMAGE_SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),
)


def _sniff_content_type(image: bytes) -> str:
    """The media type to serve `image` back as, read from its own leading
    bytes rather than from the upload's `Content-Type` header — the header
    is a claim, these are the file. Falls back to
    `application/octet-stream`, which browsers refuse to render as an
    image, making a wrong guess visible immediately instead of producing a
    silently blank editor canvas."""
    for signature, content_type in _IMAGE_SIGNATURES:
        if image.startswith(signature):
            return content_type
    logger.warning("Uploaded image matched no known signature; serving it untyped")
    return "application/octet-stream"


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


def _get_job_or_404(token: str) -> Job:
    """The job for `token`, or a 404 — shared by every route below that
    needs an existing job rather than just checking one exists."""
    job = _jobs.get(token)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    return job


def _prune_expired_jobs() -> None:
    """Drop jobs older than `JOB_TTL_SECONDS`. Called at the top of every
    request rather than on a separate background timer — this dict is
    small (one entry per in-flight mask draw across the whole bot), so an
    O(n) scan per request is cheap, and it keeps this file free of any
    asyncio-task-lifecycle bookkeeping."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for token in [token for token, job in _jobs.items() if job.created_at < cutoff]:
        del _jobs[token]


def _parse_job_meta(header_value: str | None) -> dict[str, Any]:
    """Decode the optional `X-Job-Meta` header on job creation: base64'd
    JSON, so unicode prompt text can't run into HTTP header encoding rules.
    Absent or malformed just means "no metadata, this is a plain mask job"
    — the common case, and the shape every job had before "✏️ Detail Prompt"
    started sending one."""
    if not header_value:
        return {}
    try:
        decoded = json.loads(base64.b64decode(header_value))
    except (ValueError, TypeError):
        logger.warning("Ignoring malformed X-Job-Meta header")
        return {}
    return decoded if isinstance(decoded, dict) else {}


@app.post("/jobs", dependencies=[Depends(require_shared_secret)])
async def create_job(request: Request) -> dict[str, str]:
    """comfytelegram pushes the source image here (raw bytes body) and gets
    back an unguessable token to build the WebApp button's URL from. An
    optional `X-Job-Meta` header (see `_parse_job_meta`) carries the job's
    `mode` plus, for a `mode="mask_prompt"` job, the image's current
    positive/negative prompt to pre-fill the editable fields with (the
    usual reason to open "✏️ Detail Prompt" is to cut a scene-wide prompt
    down to what the marked region needs, not type one from scratch — but
    these are only ever read back once, off `GET /jobs/{token}/result`;
    nothing is saved beyond that one submission) and read-only reference
    info to display — a plain "mask" job (the common case) sends none of
    this.

    The image is stored and later served verbatim — comfytelegram
    compresses before uploading (`handlers._to_display_jpeg`), so there is
    nothing useful left to do to these bytes here. Doing it the other way
    round, as this did originally, meant re-encoding a ~19MB PNG that had
    already spent the whole upload crossing the internet."""
    _prune_expired_jobs()
    image = await request.body()
    if not image:
        raise HTTPException(status_code=400, detail="Empty request body")
    meta = _parse_job_meta(request.headers.get("x-job-meta"))
    token = secrets.token_urlsafe(24)
    _jobs[token] = Job(
        image=image,
        image_content_type=_sniff_content_type(image),
        mode=meta.get("mode", "mask"),
        positive_prompt=meta.get("positive"),
        negative_prompt=meta.get("negative"),
        readonly_info=meta.get("readonly"),
    )
    return {"token": token}


@app.get("/jobs/{token}")
async def mask_editor_page(token: str) -> FileResponse:
    """The page Telegram opens as a WebApp — a static file, not templated;
    its own JS reads `token` back out of `location.pathname`. 404s (rather
    than silently serving the editor) once the job has expired or been
    consumed, since there'd be nothing left for it to load."""
    _prune_expired_jobs()
    _get_job_or_404(token)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/jobs/{token}/image")
async def job_image(token: str) -> Response:
    """The source image, for the editor's `<canvas>` to draw over."""
    _prune_expired_jobs()
    job = _get_job_or_404(token)
    return Response(content=job.image, media_type=job.image_content_type)


@app.get("/jobs/{token}/meta")
async def job_meta(token: str) -> JSONResponse:
    """What the editor page needs before it can even decide what to render:
    which `mode` it's in (whether to show the prompt/denoise fields next to
    the canvas), what to pre-fill them with, and read-only reference info
    to display alongside them (see `Job`). Same no-extra-auth tier as
    `/jobs/{token}/image` — the page calls this itself, before it has
    proven anything beyond holding the token."""
    _prune_expired_jobs()
    job = _get_job_or_404(token)
    return JSONResponse(
        {
            "mode": job.mode,
            "positive": job.positive_prompt,
            "negative": job.negative_prompt,
            "readonly": job.readonly_info or {},
        }
    )


def _none_if_blank(value: Any) -> str | None:
    """A submitted form field, blank-or-missing collapsed to `None` — "no
    override" rather than an empty-string override, matching how
    comfytelegram's own storage represents "use the image's own/detailer's
    own value"."""
    text = str(value).strip() if value is not None else ""
    return text or None


@app.post("/jobs/{token}/mask")
async def submit_mask(token: str, request: Request) -> dict[str, bool]:
    """The editor's "Done" button POSTs here: a multipart body with a
    `mask` file field (grayscale PNG, white = inpaint) and an `init_data`
    text field (`Telegram.WebApp.initData`, opaque to this relay — see
    module docstring for why it's forwarded rather than checked here).
    `mode="mask_prompt"` jobs also submit `positive`/`negative`/`denoise`
    text fields alongside the mask, in this same POST — one Done tap, one
    request, covering both what got drawn and what should happen there
    (any of the three may be blank, meaning "no override")."""
    _prune_expired_jobs()
    job = _get_job_or_404(token)
    form = await request.form()
    mask_file = form.get("mask")
    init_data = form.get("init_data")
    if mask_file is None or init_data is None:
        raise HTTPException(status_code=400, detail="Both 'mask' and 'init_data' are required")
    job.mask = await mask_file.read()
    if job.mode == "mask_prompt":
        job.positive_prompt = _none_if_blank(form.get("positive"))
        job.negative_prompt = _none_if_blank(form.get("negative"))
        denoise_raw = _none_if_blank(form.get("denoise"))
        job.denoise = float(denoise_raw) if denoise_raw is not None else None
    job.init_data = str(init_data)
    job.status = "submitted"
    return {"ok": True}


@app.get("/jobs/{token}/result", dependencies=[Depends(require_shared_secret)])
async def job_result(token: str) -> JSONResponse:
    """comfytelegram polls this to find out whether the editor has been
    submitted yet. Every job submits a mask, so the response shape is the
    same regardless of `mode` — `positive`/`negative`/`denoise` are simply
    `None` for a plain "mask" job. The mask itself is base64-encoded so the
    whole response can stay a single JSON body alongside `init_data`,
    rather than needing a second endpoint or a multipart response just for
    the bytes."""
    _prune_expired_jobs()
    job = _get_job_or_404(token)
    if job.status != "submitted":
        return JSONResponse({"status": "pending"})
    return JSONResponse(
        {
            "status": "submitted",
            "mask_base64": base64.b64encode(job.mask).decode("ascii"),
            "positive": job.positive_prompt,
            "negative": job.negative_prompt,
            "denoise": job.denoise,
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
