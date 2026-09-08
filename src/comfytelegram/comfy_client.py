"""Async client for a ComfyUI server's HTTP + WebSocket API.

Talks directly to a running ComfyUI instance (POST /prompt, GET /history,
GET /view, GET /object_info, WS /ws) — no dependency on the `comfy` CLI,
since the bot needs to handle many concurrent users itself.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import aiohttp


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI rejects a prompt or a job fails during execution."""


@dataclass
class JobProgress:
    prompt_id: str
    node_id: str | None
    value: int | None
    max: int | None
    done: bool = False


@dataclass
class JobResult:
    prompt_id: str
    outputs: dict[str, Any]
    """Raw `outputs` mapping from history[prompt_id]['outputs'], keyed by node id."""

    def image_refs(self) -> list[dict[str, str]]:
        """Flatten every {filename, subfolder, type} image reference across all output nodes."""
        refs: list[dict[str, str]] = []
        for node_output in self.outputs.values():
            refs.extend(node_output.get("images", []) or [])
        return refs


class ComfyClient:
    """Not thread-safe; create one per event loop. Use as an async context manager,
    or pass in a shared `aiohttp.ClientSession` you own the lifecycle of."""

    def __init__(
        self,
        http_base: str,
        ws_base: str,
        *,
        session: aiohttp.ClientSession | None = None,
    ):
        self.http_base = http_base.rstrip("/")
        self.ws_base = ws_base.rstrip("/")
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> Self:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("ComfyClient must be used as an async context manager")
        return self._session

    async def get_object_info(self) -> dict[str, Any]:
        async with self.session.get(f"{self.http_base}/object_info") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_node_info(self, class_type: str) -> dict[str, Any]:
        async with self.session.get(f"{self.http_base}/object_info/{class_type}") as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data[class_type]

    async def list_checkpoints(self) -> list[str]:
        info = await self.get_node_info("CheckpointLoaderSimple")
        return _enum_choices(info, "ckpt_name")

    async def list_loras(self) -> list[str]:
        info = await self.get_node_info("LoraLoader")
        return _enum_choices(info, "lora_name")

    async def queue_prompt(self, prompt: dict[str, Any], *, client_id: str) -> str:
        payload = {"prompt": prompt, "client_id": client_id}
        async with self.session.post(f"{self.http_base}/prompt", json=payload) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise ComfyUIError(f"ComfyUI rejected prompt: {data}")
            return data["prompt_id"]

    async def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        async with self.session.get(f"{self.http_base}/history/{prompt_id}") as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get(prompt_id)

    async def get_image_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with self.session.get(f"{self.http_base}/view", params=params) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def upload_image(
        self,
        source: Path | bytes,
        *,
        filename: str | None = None,
        subfolder: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Upload a file into ComfyUI's `input` directory, for a subsequent LoadImage.

        `source` may be a `Path` (read from disk, `filename` defaults to
        `path.name`) or raw `bytes` (`filename` required, since there's no
        path to infer one from — e.g. a bot forwarding an image it already
        holds in memory, without writing it to disk first).
        """
        if isinstance(source, Path):
            payload = source.read_bytes()
            name = filename or source.name
        else:
            if filename is None:
                raise ValueError("filename is required when uploading raw bytes")
            payload = source
            name = filename

        data = aiohttp.FormData()
        data.add_field("image", payload, filename=name)
        data.add_field("subfolder", subfolder)
        data.add_field("overwrite", "true" if overwrite else "false")
        async with self.session.post(f"{self.http_base}/upload/image", data=data) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def watch(self, prompt_id: str, *, client_id: str) -> AsyncIterator[JobProgress]:
        """Stream progress events for `prompt_id` until it's terminal.

        The final yielded event has done=True. The websocket stream itself
        doesn't carry output file references — follow up with
        `get_history(prompt_id)` for those.
        """
        url = f"{self.ws_base}/ws?clientId={client_id}"
        async with self.session.ws_connect(url) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(msg.data)
                etype = event.get("type")
                data = event.get("data", {})

                if etype == "progress" and data.get("prompt_id") == prompt_id:
                    yield JobProgress(
                        prompt_id=prompt_id,
                        node_id=data.get("node"),
                        value=data.get("value"),
                        max=data.get("max"),
                    )
                elif (
                    etype == "executing"
                    and data.get("prompt_id") == prompt_id
                    and data.get("node") is None
                ):
                    # node becomes None once the whole prompt has finished executing
                    yield JobProgress(
                        prompt_id=prompt_id, node_id=None, value=None, max=None, done=True
                    )
                    return
                elif etype == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise ComfyUIError(f"Execution error: {data}")

    async def run_and_collect(self, prompt: dict[str, Any]) -> JobResult:
        """Submit a prompt, wait for it to finish, and return its outputs.

        Convenience wrapper for callers that don't need live progress updates
        (the bot's own job runner should use `queue_prompt` + `watch` directly
        so it can push progress edits back to Telegram).
        """
        client_id = str(uuid.uuid4())
        prompt_id = await self.queue_prompt(prompt, client_id=client_id)
        async for progress in self.watch(prompt_id, client_id=client_id):
            if progress.done:
                break
        history = await self.get_history(prompt_id)
        if history is None:
            raise ComfyUIError(f"No history entry for prompt {prompt_id} after completion")
        status = history.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyUIError(f"Job {prompt_id} failed: {status}")
        return JobResult(prompt_id=prompt_id, outputs=history.get("outputs", {}))


def _enum_choices(node_info: dict[str, Any], input_name: str) -> list[str]:
    required = node_info.get("input", {}).get("required", {})
    spec = required.get(input_name)
    if not spec or not isinstance(spec[0], list):
        return []
    return list(spec[0])
