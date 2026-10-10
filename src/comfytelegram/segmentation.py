"""Image embeddings for the mask editor's "✨ Smart" select.

SAM-family segmenters split into a heavy image encoder (once per image) and
a tiny prompt/mask decoder (once per tap). The encoder runs here, on CPU via
onnxruntime, right after a mask-editor job is created (see
`handlers._relay_upload_embedding`); its output goes to inpaint_relay, and the
decoder runs in the user's browser via onnxruntime-web, so a tap never makes
a round trip.

MobileSAM rather than SAM2.1-tiny, picked by `scripts/sam_poc.py` on real
generated images: hands came out at least as clean, the encoder is ~3x
faster, and the embedding — what every phone downloads per job — is 2MB at
float16 instead of 8MB.

Model files are staged by hand at `Settings.sam_encoder_path`, same as the
WD14 tagger (see analysis.py's docstring for why nothing auto-downloads).
"""

from __future__ import annotations

import functools
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

#: SAM's input frame: the image is resized to this longest side, and the
#: browser-side decoder's points and output mask live in that frame.
SAM_INPUT_SIZE = 1024
EMBEDDING_SHAPE = (1, 256, 64, 64)


@dataclass(frozen=True)
class ImageEmbedding:
    #: Little-endian float16, `EMBEDDING_SHAPE`. Half precision halves the
    #: download for no visible change in the masks (max abs error ~2e-4).
    data: bytes
    #: The resized frame the encoder saw (see `sam_input_size`).
    width: int
    height: int


def sam_input_size(width: int, height: int) -> tuple[int, int]:
    """`(width, height)` scaled so the longest side is `SAM_INPUT_SIZE`."""
    scale = SAM_INPUT_SIZE / max(width, height)
    return max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5))


@functools.lru_cache(maxsize=1)
def _load_encoder(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def encode_image(image_bytes: bytes, encoder_path: Path) -> ImageEmbedding:
    """Run the MobileSAM encoder over `image_bytes`. Blocking (~0.3s on a
    desktop CPU) — call through `asyncio.to_thread`. This export normalizes
    and pads to 1024x1024 itself; it only wants raw 0-255 RGB at the
    resized size."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert("RGB")
    width, height = sam_input_size(*rgb.size)
    pixels = np.asarray(rgb.resize((width, height), Image.BILINEAR), dtype=np.float32)
    (embedding,) = _load_encoder(str(encoder_path)).run(None, {"input_image": pixels})
    return ImageEmbedding(embedding.astype("<f2").tobytes(), width, height)
