"""Compare OCR recognisers on real photos before porting one to OnSpeak47.

Detection is shared (RapidOCR's PP-OCRv4 det + cls), so only recognition
differs between engines:

  ppocr-ch     PP-OCRv4 Chinese rec (RapidOCR default; covers zh + en, but
               its dict lacks 123/146 Vietnamese letters)
  ppocr-latin  PP-OCRv5 Latin rec (monkt/paddleocr-onnx; dict lacks 99/146)
  vietocr      VietOCR vgg_transformer (full Vietnamese, torch, heavier)

Usage:
  drop photos into ocr_test/photos/  (optional ground truth: same name + .txt)
  ..\\..\\ConferenceASR\\.venv\\Scripts\\python.exe run_ocr_test.py
  open ocr_test/out/report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------- crops ----------------

def crop_quad(img: np.ndarray, box) -> np.ndarray:
    """PaddleOCR get_rotate_crop_image: warp the quad flat, rotate tall crops."""
    pts = np.array(box, dtype=np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(pts, dst)
    out = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_CUBIC)
    if h and w and h / w >= 1.5:
        out = np.rot90(out)
    return out


# ---------------- engines ----------------

class PPOCRLatin:
    """PP-OCRv5 latin rec, CTC decode — the same code path the phone would run."""
    name = "ppocr-latin"

    def __init__(self):
        import onnxruntime as ort
        d = HERE / "models" / "latin_v5"
        self.sess = ort.InferenceSession(str(d / "rec.onnx"),
                                         providers=["CPUExecutionProvider"])
        chars = [l.rstrip("\n") for l in open(d / "dict.txt", encoding="utf-8")]
        self.vocab = ["<blank>"] + chars + [" "]

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        h, w = crop.shape[:2]
        nw = max(8, int(np.ceil(48 * w / max(h, 1))))
        x = cv2.resize(crop, (nw, 48)).astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        probs = self.sess.run(None, {"x": x})[0][0]
        ids, conf = probs.argmax(1), probs.max(1)
        text, scores, prev = [], [], -1
        for i, c in zip(ids, conf):
            if i != prev and i != 0 and i < len(self.vocab):
                text.append(self.vocab[i])
                scores.append(c)
            prev = i
        return "".join(text), float(np.mean(scores)) if scores else 0.0


class VietOCR:
    name = "vietocr"

    def __init__(self, model: str = "vgg_transformer"):
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        cfg = Cfg.load_config_from_name(model)
        cfg["device"] = "cpu"
        cfg["cnn"]["pretrained"] = False
        self.p = Predictor(cfg)

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        text, prob = self.p.predict(Image.fromarray(crop[:, :, ::-1]), return_prob=True)
        return text, float(prob)


# ---------------- scoring ----------------

def cer(ref: str, hyp: str) -> float:
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def img_b64(img: np.ndarray, max_w: int = 900) -> str:
    pil = Image.fromarray(img[:, :, ::-1])
    if pil.width > max_w:
        pil = pil.resize((max_w, int(pil.height * max_w / pil.width)))
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default=str(HERE / "photos"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--no-vietocr", action="store_true", help="skip the torch engine")
    args = ap.parse_args()

    photos = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in IMG_EXT)
    if not photos:
        print(f"No photos in {args.photos} — drop some .jpg/.png there first.")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from rapidocr_onnxruntime import RapidOCR
    rapid = RapidOCR()
    engines = [PPOCRLatin()]
    if not args.no_vietocr:
        engines.append(VietOCR())
    names = ["ppocr-ch"] + [e.name for e in engines]

    results, totals = [], {n: 0.0 for n in names + ["det"]}
    for p in photos:
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")  # phone rotation
        img = np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])
        t0 = time.perf_counter()
        det_out, elapse = rapid(img)
        t_all = time.perf_counter() - t0
        # RapidOCR elapse = [det, cls, rec]; split det+cls from ch rec
        t_det = sum(elapse[:2]) if elapse else t_all
        totals["det"] += t_det
        totals["ppocr-ch"] += (elapse[2] if elapse else 0.0)
        det_out = det_out or []
        # reading order: top-to-bottom, then left-to-right
        det_out.sort(key=lambda r: (round(min(pt[1] for pt in r[0]) / 20), min(pt[0] for pt in r[0])))

        lines = []
        for k, (box, ch_text, ch_conf) in enumerate(det_out, 1):
            crop = crop_quad(img, box)
            row = {"id": k, "box": [[float(a), float(b)] for a, b in box],
                   "ppocr-ch": {"text": ch_text, "conf": round(float(ch_conf), 3)}}
            for e in engines:
                t = time.perf_counter()
                text, conf = e.read(crop)
                totals[e.name] += time.perf_counter() - t
                row[e.name] = {"text": text, "conf": round(conf, 3)}
            lines.append(row)

        vis = img.copy()
        for row in lines:
            pts = np.array(row["box"], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
            cv2.putText(vis, str(row["id"]), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

        entry = {"photo": p.name, "size": [pil.width, pil.height],
                 "det_ms": round(t_det * 1000), "lines": lines}
        gt_file = p.with_suffix(".txt")
        if gt_file.exists():
            gt = norm_ws(gt_file.read_text(encoding="utf-8"))
            entry["cer"] = {n: round(cer(gt, norm_ws(" ".join(r[n]["text"] for r in lines))), 3)
                            for n in names}
        entry["vis"] = img_b64(vis)
        results.append(entry)
        print(f"{p.name}: {len(lines)} lines, det {entry['det_ms']} ms"
              + (f", CER {entry['cer']}" if "cer" in entry else ""))

    n_lines = sum(len(r["lines"]) for r in results) or 1
    summary = {"photos": len(results), "lines": n_lines,
               "ms_per_photo_det": round(totals["det"] * 1000 / len(results)),
               "ms_per_line": {n: round(totals[n] * 1000 / n_lines, 1) for n in names}}
    gts = [r["cer"] for r in results if "cer" in r]
    if gts:
        summary["mean_cer"] = {n: round(sum(g[n] for g in gts) / len(gts), 3) for n in names}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    (out / "results.json").write_text(json.dumps(
        {"summary": summary, "results": [{k: v for k, v in r.items() if k != "vis"} for r in results]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out / "report.html", summary, results, names)
    print(f"report: {out / 'report.html'}")


def write_report(path: Path, summary: dict, results: list, names: list):
    e = html.escape
    parts = [f"""<!doctype html><meta charset="utf-8"><title>OCR engine test</title>
<style>
:root{{--bg:#fafafa;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--card:#fff;--acc:#7c3aed}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--card:#161b22}}}}
body{{font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px;max-width:1400px}}
h1{{font-size:20px}} h2{{font-size:16px;margin:0 0 8px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:16px}}
@media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
img{{max-width:100%;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}}
th{{color:var(--muted);font-weight:600}} .c{{color:var(--muted);font-size:11px}}
.diff{{background:color-mix(in srgb,var(--acc) 14%,transparent)}}
pre{{white-space:pre-wrap;margin:0}}
</style>
<h1>OCR engine test</h1><div class="card"><h2>Summary</h2><pre>{e(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
<p class="c">Detection is shared; per-line times are recognition only (CPU). Purple rows = engines disagree.
Ground truth: put <code>photo.txt</code> next to <code>photo.jpg</code> to get CER.</p></div>"""]
    for r in results:
        rows = []
        for ln in r["lines"]:
            texts = [ln[n]["text"] for n in names]
            cls = ' class="diff"' if len(set(map(norm_ws, texts))) > 1 else ""
            cells = "".join(f'<td>{e(ln[n]["text"])} <span class="c">{ln[n]["conf"]:.2f}</span></td>'
                            for n in names)
            rows.append(f"<tr{cls}><td>{ln['id']}</td>{cells}</tr>")
        cer_txt = f" · CER {e(json.dumps(r['cer']))}" if "cer" in r else ""
        parts.append(f"""<div class="card"><h2>{e(r['photo'])}</h2>
<p class="c">{r['size'][0]}×{r['size'][1]} · det {r['det_ms']} ms · {len(r['lines'])} lines{cer_txt}</p>
<div class="grid"><img src="data:image/jpeg;base64,{r['vis']}">
<table><tr><th>#</th>{''.join(f'<th>{n}</th>' for n in names)}</tr>{''.join(rows)}</table></div></div>""")
    path.write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
