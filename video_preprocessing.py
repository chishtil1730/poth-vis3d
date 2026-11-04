"""
PothVis3D video preprocessing.
- Detect and crop black borders (optional).
- Resize while preserving aspect ratio (no letterboxing by default).
- Optional affine video stabilization using feature-based motion estimates.
- Export normalized video and/or frames.

Example:
python video_preprocessing.py --input "C:\\Users\\shaik\\Downloads\\normalized_video.mp4" --output output/preprocessed/road_stabilized.mp4 --frames output/frames --crop-black --stabilize
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_content_crop(frame, threshold=12, margin=3):
    """Find the largest rectangle containing non-black image content."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    # Remove isolated bright noise and join nearby content.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, frame.shape[1], frame.shape[0]
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x0, y0 = max(0, x + margin), max(0, y + margin)
    x1, y1 = min(frame.shape[1], x + w - margin), min(frame.shape[0], y + h - margin)
    if x1 - x0 < frame.shape[1] * 0.5 or y1 - y0 < frame.shape[0] * 0.5:
        return 0, 0, frame.shape[1], frame.shape[0]
    return x0, y0, x1 - x0, y1 - y0


def resize_fit(frame, width, height):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


class AffineStabilizer:
    """Cumulative affine-motion smoothing. Intended for mild shake, not severe cuts."""
    def __init__(self, smoothing=20, max_corners=300):
        self.smoothing = max(1, int(smoothing))
        self.max_corners = max_corners
        self.prev_gray = None
        self.cumulative = np.zeros(3, dtype=np.float64)
        self.history = []

    def transform(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if self.prev_gray is None:
            self.prev_gray = gray
            self.history.append(self.cumulative.copy())
            return frame

        pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=self.max_corners, qualityLevel=0.01,
            minDistance=12, blockSize=7
        )
        dx = dy = da = 0.0
        if pts is not None and len(pts) >= 8:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            if nxt is not None and status is not None:
                good_old = pts[status.ravel() == 1].reshape(-1, 2)
                good_new = nxt[status.ravel() == 1].reshape(-1, 2)
                if len(good_old) >= 8:
                    M, inliers = cv2.estimateAffinePartial2D(
                        good_old, good_new, method=cv2.RANSAC,
                        ransacReprojThreshold=2.5, maxIters=1000, confidence=0.99
                    )
                    if M is not None and inliers is not None and int(inliers.sum()) >= 8:
                        dx, dy = float(M[0, 2]), float(M[1, 2])
                        da = float(np.arctan2(M[1, 0], M[0, 0]))
        self.cumulative += np.array([dx, dy, da])
        self.history.append(self.cumulative.copy())
        hist = np.asarray(self.history)
        start = max(0, len(hist) - self.smoothing)
        smooth = hist[start:].mean(axis=0)
        correction = smooth - self.cumulative
        self.prev_gray = gray
        # Limit corrections to avoid extreme warping.
        correction[0] = np.clip(correction[0], -w * 0.08, w * 0.08)
        correction[1] = np.clip(correction[1], -h * 0.08, h * 0.08)
        correction[2] = np.clip(correction[2], -0.08, 0.08)
        c, s = np.cos(correction[2]), np.sin(correction[2])
        cx, cy = w / 2, h / 2
        A = np.array([[c, -s, cx - c * cx + s * cy + correction[0]],
                      [s,  c, cy - s * cx - c * cy + correction[1]]], dtype=np.float32)
        stabilized = cv2.warpAffine(
            frame, A, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return stabilized


def process_video(input_path, output_path, frames_dir=None, target_fps=24,
                  width=512, height=512, crop_black=False, stabilize=False):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = target_fps
    writer = None
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             target_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {out_path}")
    frame_dir = Path(frames_dir) if frames_dir else None
    if frame_dir:
        frame_dir.mkdir(parents=True, exist_ok=True)

    stabilizer = AffineStabilizer() if stabilize else None
    frame_idx = saved = 0
    next_time = 0.0
    fixed_crop = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / src_fps
            frame_idx += 1
            if t + 1e-9 < next_time:
                continue
            if crop_black:
                # Estimate the crop once and reuse it; per-frame crop changes
                # would introduce artificial camera motion into visual odometry.
                if fixed_crop is None:
                    fixed_crop = detect_content_crop(frame)
                x, y, w, h = fixed_crop
                frame = frame[y:y + h, x:x + w]
            frame = resize_fit(frame, width, height)
            if stabilizer:
                frame = stabilizer.transform(frame)
            writer.write(frame)
            if frame_dir:
                name = f"frame_{saved:06d}.jpg"
                if not cv2.imwrite(str(frame_dir / name), frame):
                    raise RuntimeError(f"Could not save frame {name}")
            saved += 1
            next_time = saved / target_fps
    finally:
        cap.release()
        writer.release()
    print(f"Input frames read: {frame_idx}")
    print(f"Output frames: {saved}")
    print(f"Output video: {out_path.resolve()}")
    if frame_dir:
        print(f"Frames folder: {frame_dir.resolve()}")
    print("Note: stabilization is a 2D visual correction; it does not recover physical camera motion.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crop, normalize and optionally stabilize road video.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="output/preprocessed/road_stabilized.mp4")
    ap.add_argument("--frames", default="output/frames")
    ap.add_argument("--fps", type=float, default=24)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--crop-black", action="store_true")
    ap.add_argument("--stabilize", action="store_true")
    a = ap.parse_args()
    if a.fps <= 0 or a.width <= 0 or a.height <= 0:
        ap.error("fps, width and height must be positive")
    process_video(a.input, a.output, a.frames, a.fps, a.width, a.height,
                  a.crop_black, a.stabilize)
