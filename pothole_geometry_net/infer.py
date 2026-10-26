"""Run on a frame folder/image. Example:
python infer.py --ckpt outputs/run1/best.pt --input C:/Users/LENOVO/Dsa/Potholes/output/frames --out C:/Users/LENOVO/Dsa/Potholes/output/depth_results"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import cv2, numpy as np, torch
from dataset import letterbox, to_tensor
from model import PotholeGeometryNet
from utils import colorize, hstack


def regions(prob, dep, logvar, thr=.5, min_area=30) -> list[dict]:
    n, lab, st, _ = cv2.connectedComponentsWithStats((prob > thr).astype(np.uint8), connectivity=8); res = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        if area < min_area: continue
        m = lab == i; d = dep[m]
        res.append({"area_px": area, "bbox_xywh": [x, y, w, h], "median_depression": float(np.median(d)), "p90_depression": float(np.percentile(d, 90)),
                    "max_depression": float(d.max()), "mean_log_variance": float(logvar[m].mean()), "confidence": float(prob[m].mean())})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--input", required=True); ap.add_argument("--out", default="outputs/infer")
    ap.add_argument("--size", type=int, default=512); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--thr", type=float, default=.5); ap.add_argument("--no-panels", action="store_true")
    a = ap.parse_args(); dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    ck = torch.load(a.ckpt, map_location=dev); model = PotholeGeometryNet(tuple(ck["config"]["model"]["channels"]))
    model.load_state_dict(ck["model_state_dict"]); model.to(dev).eval(); half = a.fp16 and dev.type == "cuda"
    if half: model.half()
    p = Path(a.input); files = [p] if p.is_file() else sorted((f for f in p.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}),
                                                              key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f.name)])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); summary = {}
    for i in range(0, len(files), a.batch):
        chunk, rgbs, metas = files[i:i + a.batch], [], []
        for f in chunk:
            lb, meta = letterbox(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB), a.size); rgbs.append(lb); metas.append(meta)
        x = torch.stack([to_tensor(r) for r in rgbs]).to(dev); x = x.half() if half else x
        with torch.inference_mode():
            o = model(x)
        prob = torch.sigmoid(o["pothole_logits"].float())[:, 0].cpu().numpy(); dep = o["depression"].float()[:, 0].cpu().numpy(); lv = o["log_variance"].float()[:, 0].cpu().numpy()
        for j, f in enumerate(chunk):
            g = rgbs[j].max(2) > 5  # real image content (excludes black letterbox bars)
            ok = g.any(1)[:, None] & g.any(0)[None, :]
            prob[j][~ok] = 0; dep[j][~ok] = 0; lv[j][~ok] = 0
            s = f.stem; np.save(out / f"{s}_depression.npy", dep[j]); np.save(out / f"{s}_logvar.npy", lv[j]); np.save(out / f"{s}_prob.npy", prob[j].astype(np.float16))
            r = regions(prob[j], dep[j], lv[j], a.thr); summary[f.name] = {"meta": metas[j], "units": "normalized relative (uncalibrated)", "potholes": r}
            if not a.no_panels:
                vmax = max(float(dep[j].max()), 1e-3)
                cv2.imwrite(str(out / f"{s}_panel.png"), hstack([rgbs[j][..., ::-1], (prob[j] * 255).astype(np.uint8), colorize(dep[j], 0, vmax), colorize(np.exp(.5 * lv[j]), 0, float(np.exp(.5 * lv[j]).max()) + 1e-6)]))
    json.dump(summary, open(out / "regions.json", "w"), indent=1); print(f"{len(files)} images -> {out}")


if __name__ == "__main__":
    main()