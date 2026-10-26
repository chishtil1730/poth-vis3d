"""Live pothole pipeline. Put in the Potholes/ folder (next to scene_tracking.py).

  python live_pipeline.py --source 0                      # webcam
  python live_pipeline.py --source rtsp://user:pw@ip/stream
  python live_pipeline.py --source road.mp4               # file, paced like live

Priority: detection + tracking every frame (main loop) > depth (async thread, best effort)
> reconstruction (optional, asked at exit, uses recorded tracks).
Keys: q quit | [ ] depth opacity | d depth on/off | t track overlay on/off | s save snapshot
"""
import argparse, json, sys, threading, time
from pathlib import Path
import cv2, numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "pothole_geometry_net"))
import scene_tracking as st                       # reuse your tracking functions
import pothole_detection as pd                    # reuse your YOLO loader
from model import PotholeGeometryNet
from dataset import letterbox, to_tensor


class Grabber(threading.Thread):
    """Reads the source continuously, keeps only the newest frame (no lag buildup)."""
    def __init__(self, src, pace):
        super().__init__(daemon=True)
        src = int(src) if str(src).isdigit() else src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")
        self.is_file = isinstance(src, str) and Path(src).exists()
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.delay = (1 / fps) if (self.is_file and pace) else 0
        self.frame, self.id, self.ok, self.lock = None, 0, True, threading.Lock()

    def run(self):
        while self.ok:
            t0 = time.time()
            good, f = self.cap.read()
            if not good:
                self.ok = False; break
            with self.lock:
                self.frame, self.id = f, self.id + 1
            if self.delay:
                time.sleep(max(0, self.delay - (time.time() - t0)))

    def latest(self, last_id):
        with self.lock:
            return (self.frame, self.id) if self.id != last_id else (None, last_id)


class DepthWorker(threading.Thread):
    """Runs PotholeGeometryNet on the newest frame whenever the GPU is free (rate-limited)."""
    def __init__(self, ckpt, dev, size, interval):
        super().__init__(daemon=True)
        ck = torch.load(ckpt, map_location=dev)
        self.m = PotholeGeometryNet(tuple(ck["config"]["model"]["channels"]))
        self.m.load_state_dict(ck["model_state_dict"]); self.m.to(dev).eval()
        self.half = dev.type == "cuda"
        if self.half: self.m.half()
        self.dev, self.size, self.interval = dev, size, interval
        self.inp, self.out, self.enabled, self.ok = None, None, True, True
        self.ms, self.lock = 0.0, threading.Lock()

    def submit(self, frame):
        with self.lock: self.inp = frame

    def run(self):
        while self.ok:
            time.sleep(self.interval)
            with self.lock: f, self.inp = self.inp, None
            if f is None or not self.enabled: continue
            t0 = time.time()
            lb, meta = letterbox(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), self.size)
            x = to_tensor(lb)[None].to(self.dev)
            x = x.half() if self.half else x
            with torch.inference_mode():
                o = self.m(x)
            dep = o["depression"].float()[0, 0].cpu().numpy()
            py, px, s = meta["pad_y"], meta["pad_x"], meta["scale"]
            h, w = meta["orig_hw"]
            dep = dep[py:py + round(h * s), px:px + round(w * s)]
            dep = cv2.resize(dep, (w, h))
            self.out, self.ms = dep, (time.time() - t0) * 1000


def det_mask_and_draw(res, shape, canvas):
    """Pothole mask (255=pothole) for the tracker + overlay drawn on canvas."""
    mask = np.zeros(shape[:2], np.uint8); n = 0
    if res.boxes is not None and len(res.boxes):
        boxes, conf = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
        polys = res.masks.xy if res.masks is not None else []
        ov = canvas.copy()
        for i, p in enumerate(polys):
            if len(p) >= 3:
                cv2.fillPoly(mask, [p.astype(np.int32)], 255)
                cv2.fillPoly(ov, [p.astype(np.int32)], (0, 0, 255))
        cv2.addWeighted(ov, 0.35, canvas, 0.65, 0, canvas)
        for b, c in zip(boxes, conf):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(canvas, f"pothole {c:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        n = len(boxes)
    return mask, n


def depth_overlay(canvas, dep, alpha=0.85):
    v = max(float(dep.max()), 1e-3)
    if v < 0.02: return canvas                      # nothing meaningful
    cm = cv2.applyColorMap((np.clip(dep / v, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = (np.clip(dep / v, 0, 1) ** 0.5 * alpha)[..., None]   # sqrt boosts faint values; alpha = max opacity
    return (canvas * (1 - a) + cm * a).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=640, help="processing width (height keeps aspect)")
    ap.add_argument("--imgsz", type=int, default=512, help="YOLO size")
    ap.add_argument("--ckpt", default=str(HERE / "pothole_geometry_net/outputs/run1/best.pt"))
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--depth-alpha", type=float, default=0.85, help="max depth overlay opacity 0-1")
    ap.add_argument("--depth-interval", type=float, default=0.3, help="min seconds between depth runs")
    ap.add_argument("--no-pace", action="store_true", help="video files: run as fast as possible")
    ap.add_argument("--out", default="output")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--max-track-records", type=int, default=6000)
    a = ap.parse_args()

    out = Path(a.out); (out / "scene_tracking").mkdir(parents=True, exist_ok=True)
    pd.IMAGE_SIZE = a.imgsz
    model, device = pd.load_model()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth = None
    if not a.no_depth and Path(a.ckpt).exists():
        depth = DepthWorker(a.ckpt, dev, 512, a.depth_interval); depth.start()
    elif not a.no_depth:
        print(f"Depth checkpoint missing ({a.ckpt}); depth disabled.")

    g = Grabber(a.source, not a.no_pace); g.start()
    prev_gray, points, next_id, last_id = None, [], 0, 0
    records, show_depth, show_tracks, alpha, size = [], True, True, a.depth_alpha, None
    writer, fps, t_prev, fidx = None, 0.0, time.time(), 0
    cv2.namedWindow("Pothole Live", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Depth opacity %", "Pothole Live", int(a.depth_alpha * 100), 100, lambda v: None)

    while True:
        frame, last_id2 = g.latest(last_id)
        if frame is None:
            if not g.ok: break
            time.sleep(0.002); continue
        dropped = last_id2 - last_id - 1; last_id = last_id2
        h0, w0 = frame.shape[:2]
        if w0 != a.width:
            frame = cv2.resize(frame, (a.width, round(h0 * a.width / w0)))
        size = frame.shape[:2]
        t0 = time.time()

        # 1) detection (GPU)
        res = model.predict(frame, imgsz=a.imgsz, conf=pd.CONFIDENCE, iou=pd.IOU_THRESHOLD,
                            max_det=pd.MAX_DETECTIONS, device=device, half=(device != "cpu"), verbose=False)[0]
        canvas = frame.copy()
        alpha = cv2.getTrackbarPos("Depth opacity %", "Pothole Live") / 100
        if depth and depth.enabled:
            depth.submit(frame)
            if depth.out is not None and depth.out.shape[:2] == canvas.shape[:2] and show_depth:
                canvas = depth_overlay(canvas, depth.out, alpha)
        mask, n_pot = det_mask_and_draw(res, frame.shape, canvas)
        t_det = (time.time() - t0) * 1000

        # 2) tracking (CPU, your functions)
        t1 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = {"valid": False, "model": None, "inliers": 0, "matrix": None}
        if prev_gray is None or prev_gray.shape != gray.shape:
            points, next_id = st.detect_new_points(gray, mask, [], next_id)
        else:
            points, corr = st.track_points(prev_gray, gray, points)
            old_p, new_p = corr if len(corr) == 2 else ([], [])
            motion = st.estimate_camera_motion(old_p, new_p)
            points, next_id = st.detect_new_points(gray, mask, points, next_id)
        prev_gray = gray
        if show_tracks:
            canvas = st.draw_tracks(canvas, points, fidx, motion)
        records.append({"frame": f"live_{fidx:06d}", "frame_index": fidx,
                        "tracks": [{"id": p["id"], "x": round(float(p["point"][0]), 2),
                                    "y": round(float(p["point"][1]), 2), "category": p["category"]} for p in points]})
        if len(records) > a.max_track_records: records.pop(0)
        t_trk = (time.time() - t1) * 1000

        # HUD
        now = time.time(); fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6); t_prev = now
        hud = (f"FPS {fps:4.1f} | det {t_det:3.0f}ms trk {t_trk:3.0f}ms"
               + (f" depth {depth.ms:3.0f}ms" if depth and depth.enabled else " depth off")
               + f" | potholes {n_pot} | pts {len(points)} | drop {dropped}")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(canvas, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        cv2.imshow("Pothole Live", canvas)
        if a.save_video:
            if writer is None:
                writer = cv2.VideoWriter(str(out / "live_result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 18,
                                         (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        fidx += 1

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        elif k == ord("d") and depth: depth.enabled = not depth.enabled
        elif k == ord("]"): cv2.setTrackbarPos("Depth opacity %", "Pothole Live", min(100, int(alpha * 100) + 10))
        elif k == ord("["): cv2.setTrackbarPos("Depth opacity %", "Pothole Live", max(0, int(alpha * 100) - 10))
        elif k == ord("t"): show_tracks = not show_tracks
        elif k == ord("s"): cv2.imwrite(str(out / f"snap_{fidx:06d}.png"), canvas)

    g.ok = False
    if depth: depth.ok = False
    if writer: writer.release()
    cv2.destroyAllWindows()

    # 3) optional reconstruction, chosen at the end
    if len(records) > 30 and input(f"\nGenerate camera poses/sparse points from {len(records)} frames? [y/N] ").lower() == "y":
        tp = out / "scene_tracking" / "tracks.json"
        json.dump(records, open(tp, "w"))
        import camera_pose_generator as cpg
        h, w = size
        sys.argv = ["camera_pose_generator", "--tracks", str(tp), "--out", str(out / "reconstruction"),
                    "--width", str(w), "--height", str(h)]
        cpg.main()


if __name__ == "__main__":
    main()