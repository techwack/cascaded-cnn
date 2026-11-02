"""Live demo: run held-out test images (or your own) through the trained cascade.

    python demo/app.py                      # checkpoints from the config's out_dir
    python demo/app.py --run-dir runs_final # or any folder with all five nodes
    open http://127.0.0.1:8000

Shows the decision and class probabilities at every stage, and which stages
were skipped by the early exit. Binds to localhost only.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from solarnet.common import (
    load_cascade,
    load_config,
    preprocess,
    to_batch,
)
from solarnet.data import IMG_H, IMG_W, load_image, load_records
from solarnet.taxonomy import Taxonomy

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
DISPLAY_SCALE = 8
BATCH = 256
# Display-only colour map (thermal look); the models see the raw grayscale values.
_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_COLORS = np.array(
    [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)],
    dtype=np.float32,
)
LUT = np.stack(
    [np.interp(np.arange(256) / 255.0, _STOPS, _COLORS[:, c]) for c in range(3)], axis=1
).astype(np.uint8)


def to_data_uri(img: np.ndarray) -> str:
    rgb = LUT[(img.clip(0, 1) * 255).astype(np.uint8)]
    big = Image.fromarray(rgb).resize(
        (IMG_W * DISPLAY_SCALE, IMG_H * DISPLAY_SCALE), Image.NEAREST
    )
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Demo:
    def __init__(self, config: str, run_dir: str | None, taxonomy: str):
        self.cfg = load_config(config)
        self.tax = Taxonomy.load(taxonomy)
        self.net = load_cascade(
            self.cfg, self.tax, Path(run_dir or self.cfg["train"]["out_dir"])
        )
        saved = json.load(open(self.cfg["data"]["splits_file"]))
        self.stats, self.up = saved["stats"], self.cfg["data"]["upsample"]

        by_id = {r["id"]: r for r in load_records(self.cfg["data"]["root"])}
        self.test = [by_id[i] for i in saved["test"]]
        imgs = np.stack([load_image(r["path"]) for r in self.test])
        pred, exited = self._predict_all(imgs)
        truth = np.array([self.tax.index(r["label"]) for r in self.test])
        self.accuracy = float((pred == truth).mean())
        self.early_exit = float(exited.mean())
        self.wrong = [r for r, p, t in zip(self.test, pred, truth) if p != t]
        self.by_class = {
            c: [r for r in self.test if r["label"] == c] for c in self.tax.classes
        }

    def _predict_all(self, imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds, exited = [], []
        for i in range(0, len(imgs), BATCH):
            x = preprocess(to_batch(imgs[i : i + BATCH]), self.stats, self.up)
            out = self.net.predict(x)
            preds.append(out["pred"])
            exited.append(~out["is_defect"])
        return torch.cat(preds).numpy(), torch.cat(exited).numpy()

    @torch.inference_mode()
    def trace(self, img: np.ndarray) -> dict:
        """Run one image through the cascade, recording every stage's probabilities."""
        x = preprocess(to_batch(img[None]), self.stats, self.up)
        t0 = time.perf_counter()
        p1 = torch.softmax(self.net.triage(x), 1)[0]
        stages = [
            {
                "title": "Stage 1 · Triage",
                "classes": ["No-Anomaly", "Defect"],
                "probs": p1.tolist(),
                "pick": int(p1.argmax()),
            }
        ]
        if stages[0]["pick"] == 0:
            final = self.tax.healthy
        else:
            p2 = torch.softmax(self.net.family(x), 1)[0]
            fam = self.tax.family_names[int(p2.argmax())]
            stages.append(
                {
                    "title": "Stage 2 · Fault family",
                    "classes": self.tax.family_names,
                    "probs": p2.tolist(),
                    "pick": int(p2.argmax()),
                }
            )
            p3 = torch.softmax(self.net.fine[fam](x), 1)[0]
            classes3 = list(self.tax.families[fam])
            stages.append(
                {
                    "title": f"Stage 3 · {fam} specialist",
                    "classes": classes3,
                    "probs": p3.tolist(),
                    "pick": int(p3.argmax()),
                }
            )
            final = classes3[int(p3.argmax())]
        return {
            "image": to_data_uri(img),
            "stages": stages,
            "prediction": final,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    def sample(self, cls: str | None, mistake: bool) -> dict:
        if mistake:
            rec = random.choice(self.wrong)
        elif cls and cls in self.by_class:
            rec = random.choice(self.by_class[cls])
        else:
            rec = random.choice(self.test)
        out = self.trace(load_image(rec["path"]))
        return {
            **out,
            "truth": rec["label"],
            "correct": out["prediction"] == rec["label"],
            "id": rec["id"],
        }

    def upload(self, data: bytes) -> dict:
        img = Image.open(io.BytesIO(data)).convert("L")
        # BOX averages each source block exactly, so an enlarged dataset image maps back to its original pixels.
        shrinking = img.width >= IMG_W and img.height >= IMG_H
        img = img.resize((IMG_W, IMG_H), Image.BOX if shrinking else Image.BILINEAR)
        out = self.trace(np.asarray(img, dtype=np.float32) / 255.0)
        return {**out, "truth": None, "correct": None, "id": "upload"}

    def meta(self) -> dict:
        return {
            "classes": self.tax.classes,
            "accuracy": self.accuracy,
            "early_exit": self.early_exit,
            "n_test": len(self.test),
            "n_wrong": len(self.wrong),
        }


def make_handler(demo: Demo, page: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: dict, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif url.path == "/api/meta":
                self._json(demo.meta())
            elif url.path == "/api/sample":
                cls = q.get("cls", [None])[0]
                self._json(demo.sample(cls, q.get("mode", [""])[0] == "mistake"))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/upload":
                return self._json({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= MAX_UPLOAD_BYTES:
                return self._json({"error": "image must be under 2 MB"}, 400)
            try:
                self._json(demo.upload(self.rfile.read(n)))
            except Exception:
                self._json({"error": "could not read that file as an image"}, 400)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/improved3.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="checkpoint folder (default: the config's out_dir)",
    )
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print("loading cascade and test split ...", flush=True)
    demo = Demo(args.config, args.run_dir, args.taxonomy)
    page = (Path(__file__).parent / "index.html").read_bytes()
    print(
        f"test accuracy {demo.accuracy:.4f} on {len(demo.test)} images; serving http://127.0.0.1:{args.port}",
        flush=True,
    )
    ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(demo, page)
    ).serve_forever()


if __name__ == "__main__":
    main()
