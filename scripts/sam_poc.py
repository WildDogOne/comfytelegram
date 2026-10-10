"""Proof of concept: point-prompted segmentation (SAM) on CPU via onnxruntime.

Not part of the bot or the pytest suite. Checks whether a SAM-family model is
good and fast enough for a "smart select" brush in inpaint_relay's mask editor,
where the plan is: encode the image once on the bot host, upload the embedding
to the relay, and run only the small decoder in the browser (onnxruntime-web).
So this times the two halves separately and reports the embedding's size, the
thing every phone would have to download per job.

Models (download into --models-dir):
  sam21t/     https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX
              (onnx/vision_encoder.onnx[_data], onnx/prompt_encoder_mask_decoder.onnx[_data])
  mobilesam/  https://huggingface.co/Acly/MobileSAM
              (mobile_sam_image_encoder.onnx, sam_mask_decoder_multi.onnx)

    uv run python scripts/sam_poc.py IMAGE --models-dir DIR --out-dir OUT \\
        --pos 500,375 [--pos x,y ...] [--neg x,y ...]

Points are in the original image's pixel coordinates.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

SIZE = 1024


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _points(pos: list[tuple[float, float]], neg: list[tuple[float, float]]):
    coords = np.array(pos + neg, dtype=np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    return coords, labels


def run_sam21(models: Path, img: Image.Image, coords, labels, decode_runs: int):
    enc = _session(models / "sam21t/onnx/vision_encoder.onnx")
    dec = _session(models / "sam21t/onnx/prompt_encoder_mask_decoder.onnx")
    w, h = img.size
    # HF's Sam2ImageProcessor squashes to 1024x1024 (no aspect-preserving pad).
    x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)

    t = time.perf_counter()
    emb = enc.run(None, {"pixel_values": x})
    t_enc = time.perf_counter() - t

    pts = coords * [SIZE / w, SIZE / h]
    feeds = {
        "input_points": pts[None, None].astype(np.float32),
        "input_labels": labels[None, None].astype(np.int64),
        "input_boxes": np.zeros((1, 0, 4), dtype=np.float32),
        "image_embeddings.0": emb[0],
        "image_embeddings.1": emb[1],
        "image_embeddings.2": emb[2],
    }
    t = time.perf_counter()
    for _ in range(decode_runs):
        iou, masks, _obj = dec.run(None, feeds)
    t_dec = (time.perf_counter() - t) / decode_runs

    # Each candidate is 256x256 logits over the squashed 1024 square.
    cands = [
        np.asarray(Image.fromarray(m.astype(np.float32)).resize((w, h), Image.BILINEAR)) > 0
        for m in masks[0, 0]
    ]
    emb_bytes = sum(e.size for e in emb) * 2  # as float16
    return cands, int(np.argmax(iou[0, 0])), t_enc, t_dec, emb_bytes, iou[0, 0].tolist()


def run_mobilesam(models: Path, img: Image.Image, coords, labels, decode_runs: int):
    enc = _session(models / "mobilesam/mobile_sam_image_encoder.onnx")
    dec = _session(models / "mobilesam/sam_mask_decoder_multi.onnx")
    w, h = img.size
    # Original SAM preprocessing: longest side to 1024, keep aspect; the
    # encoder export normalizes and pads to 1024x1024 itself.
    s = SIZE / max(w, h)
    rw, rh = round(w * s), round(h * s)
    x = np.asarray(img.resize((rw, rh), Image.BILINEAR), dtype=np.float32)

    t = time.perf_counter()
    (emb,) = enc.run(None, {"input_image": x})
    t_enc = time.perf_counter() - t

    # No box prompt -> SAM expects one padding point labelled -1.
    pts = np.concatenate([coords * s, [[0.0, 0.0]]]).astype(np.float32)
    lbl = np.concatenate([labels, [-1]]).astype(np.float32)
    feeds = {
        "image_embeddings": emb,
        "point_coords": pts[None],
        "point_labels": lbl[None],
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([h, w], dtype=np.float32),
    }
    t = time.perf_counter()
    for _ in range(decode_runs):
        masks, iou, _low = dec.run(None, feeds)
    t_dec = (time.perf_counter() - t) / decode_runs

    # Mask 0 is the "single point is ambiguous" fallback; SAM's own predictor
    # picks the best of 1..3 when there's more than one point.
    cand = range(4) if len(coords) == 1 else range(1, 4)
    best = max(cand, key=lambda i: iou[0, i])
    return list(masks[0] > 0), best, t_enc, t_dec, emb.size * 2, iou[0].tolist()


def overlay(img: Image.Image, mask: np.ndarray, coords, labels, path: Path) -> None:
    base = np.asarray(img, dtype=np.float32)
    red = np.array([255, 45, 45], dtype=np.float32)
    out = np.where(mask[..., None], base * 0.4 + red * 0.6, base).astype(np.uint8)
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    r = max(4, round(max(img.size) / 150))
    for (x, y), lab in zip(coords, labels):
        d.ellipse((x - r, y - r, x + r, y + r), fill="lime" if lab else "blue", outline="black")
    im.save(path)


def _xy(s: str) -> tuple[float, float]:
    x, y = s.split(",")
    return float(x), float(y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pos", type=_xy, action="append", default=[])
    ap.add_argument("--neg", type=_xy, action="append", default=[])
    ap.add_argument("--model", choices=["sam21t", "mobilesam", "both"], default="both")
    ap.add_argument("--decode-runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    ap.add_argument("--all-masks", action="store_true", help="also save every candidate mask")
    args = ap.parse_args()
    if not args.pos:
        ap.error("need at least one --pos point")
    if args.threads:
        # Read by _session via a module-level default; simplest for a PoC.
        global _session
        n = args.threads

        def _session(path: Path) -> ort.InferenceSession:
            so = ort.SessionOptions()
            so.intra_op_num_threads = n
            so.inter_op_num_threads = 1
            return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    img = Image.open(args.image).convert("RGB")
    coords, labels = _points(args.pos, args.neg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runners = {"sam21t": run_sam21, "mobilesam": run_mobilesam}
    names = list(runners) if args.model == "both" else [args.model]
    print(f"{args.image.name}: {img.size[0]}x{img.size[1]}, {len(args.pos)}+/{len(args.neg)}-")
    for name in names:
        cands, best, t_enc, t_dec, emb_bytes, iou = runners[name](
            args.models_dir, img, coords, labels, args.decode_runs
        )
        mask = cands[best]
        out = args.out_dir / f"{args.image.stem}_{name}.png"
        overlay(img, mask, coords, labels, out)
        if args.all_masks:
            for i, m in enumerate(cands):
                overlay(img, m, coords, labels, out.with_name(f"{out.stem}_c{i}.png"))
        print(
            f"  {name:9s} encode {t_enc * 1000:7.0f} ms  decode {t_dec * 1000:6.1f} ms  "
            f"embedding(fp16) {emb_bytes / 1e6:5.1f} MB  "
            f"coverage {mask.mean() * 100:5.1f}%  iou {[round(v, 2) for v in iou]}  -> {out}"
        )


if __name__ == "__main__":
    main()
