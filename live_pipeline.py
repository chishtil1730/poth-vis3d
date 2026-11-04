
"""
PothVis3D live pothole pipeline.

Usage:
    python live_pipeline.py --source 0
    python live_pipeline.py --source road.mp4
    python live_pipeline.py --source rtsp://user:pw@ip/stream

Display controls:
    1 = Bounding boxes
    2 = Segmentation
    3 = Depression heatmap
    t = Toggle feature tracks
    d = Enable/disable depth worker
    s = Save snapshot
    q = Quit
"""

import argparse
import json
import sys
import threading
import time

from pathlib import Path

import cv2
import numpy as np
import torch


# --------------------------------------------------
# PROJECT PATHS AND IMPORTS
# --------------------------------------------------

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "pothole_geometry_net"))

import scene_tracking as st
import pothole_detection as pd

from depth_anything_v2.dpt import DepthAnythingV2


# --------------------------------------------------
# VIDEO FRAME GRABBER
# --------------------------------------------------

class Grabber(threading.Thread):

    def __init__(self, src, pace=True):

        super().__init__(daemon=True)

        src = int(src) if str(src).isdigit() else src

        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        self.is_file = (
            isinstance(src, str)
            and Path(src).exists()
        )

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        self.delay = (
            1 / fps
            if self.is_file and pace
            else 0
        )

        self.frame = None
        self.id = 0
        self.ok = True
        self.lock = threading.Lock()

    def run(self):

        while self.ok:

            t0 = time.time()

            good, frame = self.cap.read()

            if not good:
                self.ok = False
                break

            with self.lock:
                self.frame = frame
                self.id += 1

            if self.delay:
                time.sleep(
                    max(0, self.delay - (time.time() - t0))
                )

    def latest(self, last_id):

        with self.lock:

            if self.id != last_id:
                return self.frame, self.id

            return None, last_id

    def release(self):

        self.ok = False
        self.cap.release()


# --------------------------------------------------
# DEPTH / DEPRESSION WORKER
# --------------------------------------------------

class DepthWorker(threading.Thread):

    def __init__(
        self,
        ckpt,
        dev,
        size=518,
        interval=0.1
    ):

        super().__init__(daemon=True)

        self.model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384]
        )

        checkpoint = torch.load(
            ckpt,
            map_location=dev,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint)
        self.model.to(dev).eval()

        self.half = (dev.type == "cuda")
        if self.half:
            self.model.half()

        self.dev = dev
        self.size = size
        self.interval = interval

        self.inp = None
        self.out = None
        self.out_frame = None
        self.out_fidx = -1

        self.enabled = True
        self.ok = True

        self.ms = 0.0
        self.lock = threading.Lock()

    def submit(self, frame, mask=None, fidx=0):

        with self.lock:
            self.inp = (
                frame.copy(),
                mask.copy() if mask is not None else None,
                fidx
            )

    @staticmethod
    def extract_relative_depression(depth, mask, fidx=None):
        """
        Calculates relative pothole depression from Depth Anything V2 scene depth.
        Uses the surrounding local road surface ring to find the baseline road depth.
        Depression is strictly calculated inside the YOLO pothole segmentation regions.
        Outside potholes, depression is strictly 0.0.
        """
        if depth is None:
            return None

        if mask is None or not np.any(mask > 0):
            return np.zeros_like(depth, dtype=np.float32)

        h, w = depth.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        binary_mask = (mask > 127).astype(np.uint8)
        if not np.any(binary_mask):
            return np.zeros_like(depth, dtype=np.float32)

        depression_map = np.zeros_like(depth, dtype=np.float32)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        for label_id in range(1, num_labels):
            pothole_roi = (labels == label_id).astype(np.uint8)
            if stats[label_id, cv2.CC_STAT_AREA] < 10:
                continue

            # 1. Local road ring excluding ALL potholes
            dilated = cv2.dilate(pothole_roi, kernel, iterations=2)
            road_ring = cv2.bitwise_and(dilated, cv2.bitwise_not(binary_mask))

            # 2. Local road baseline disparity (strictly finite, valid values)
            ring_pixels = depth[road_ring > 0]
            valid_ring = ring_pixels[np.isfinite(ring_pixels) & (ring_pixels > 0)]

            if len(valid_ring) >= 10:
                road_ref = float(np.median(valid_ring))
            else:
                x, y, bw, bh = stats[label_id, :4]
                pad = 20
                y1, y2 = max(0, y - pad), min(h, y + bh + pad)
                x1, x2 = max(0, x - pad), min(w, x + bw + pad)
                patch_mask = binary_mask[y1:y2, x1:x2]
                patch_depth = depth[y1:y2, x1:x2]
                valid_patch = patch_depth[
                    (patch_mask == 0)
                    & np.isfinite(patch_depth)
                    & (patch_depth > 0)
                ]

                if len(valid_patch) >= 10:
                    road_ref = float(np.median(valid_patch))
                else:
                    frame_str = f"Frame {fidx}: " if fidx is not None else ""
                    print(
                        f"[Warning] {frame_str}Unable to compute reliable local road baseline "
                        f"for pothole #{label_id}. Skipping to avoid false depression."
                    )
                    continue

            pothole_depth = depth[pothole_roi > 0]
            local_delta = np.clip(road_ref - pothole_depth, 0.0, None)

            depression_map[pothole_roi > 0] = local_delta

        return depression_map

    def get_latest(self):
        """Return a completed depth map together with its exact source frame."""
        with self.lock:
            if self.out is None or self.out_frame is None:
                return None, None, -1

            return (
                self.out.copy(),
                self.out_frame.copy(),
                self.out_fidx
            )

    def run(self):

        while self.ok:

            time.sleep(self.interval)

            with self.lock:
                task = self.inp
                self.inp = None

            if task is None or not self.enabled:
                continue

            frame, mask, fidx = task
            t0 = time.time()

            try:
                raw_depth = self.model.infer_image(
                    frame,
                    input_size=self.size
                )

                depression = self.extract_relative_depression(
                    raw_depth,
                    mask,
                    fidx
                )
                if depression is not None:
                   active = depression[depression > 1e-4]
                   print(
                       f"[DEPTH RESULT] Frame={fidx}, "
                       f"Positive pixels={active.size}, "
                       f"Maximum={float(active.max()) if active.size else 0:.5f}"
                    )

                with self.lock:
                    self.out = depression
                    self.out_frame = frame.copy()
                    self.out_fidx = fidx

                self.ms = (time.time() - t0) * 1000

            except Exception as error:
                print(f"Depth Anything V2 worker error: {error}")

    def stop(self):

        self.ok = False


# --------------------------------------------------
# CREATE INTERNAL POTHOLE MASK
# Used for tracking, not displayed as a separate mode.
# --------------------------------------------------

def build_pothole_mask(result, shape):

    mask = np.zeros(
        shape[:2],
        dtype=np.uint8
    )

    pothole_count = 0

    if result.boxes is not None and len(result.boxes):

        pothole_count = len(result.boxes)

        if result.masks is not None:

            for polygon in result.masks.xy:

                if len(polygon) >= 3:

                    cv2.fillPoly(
                        mask,
                        [polygon.astype(np.int32)],
                        255
                    )

    return mask, pothole_count


# --------------------------------------------------
# RENDER BOUNDING BOXES
# --------------------------------------------------

def render_boxes(frame, result):

    view = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return view

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for box, confidence in zip(boxes, confidences):

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            view,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            view,
            f"pothole {confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    return view


# --------------------------------------------------
# RENDER SEGMENTATION
# --------------------------------------------------

def render_segmentation(frame, result):

    view = frame.copy()

    if result.masks is None:
        return view

    overlay = frame.copy()

    for polygon in result.masks.xy:

        if len(polygon) < 3:
            continue

        points = polygon.astype(np.int32)

        cv2.fillPoly(
            overlay,
            [points],
            (0, 0, 255)
        )

    view = cv2.addWeighted(
        overlay,
        0.35,
        frame,
        0.65,
        0
    )

    for polygon in result.masks.xy:

        if len(polygon) >= 3:

            cv2.polylines(
                view,
                [polygon.astype(np.int32)],
                True,
                (0, 255, 255),
                2
            )

    return view


# --------------------------------------------------
# RENDER DEPRESSION HEATMAP
# --------------------------------------------------

def depth_overlay(frame, depression, alpha=0.85):

    if depression is None:
        return frame.copy()

    if depression.shape[:2] != frame.shape[:2]:
        return frame.copy()

    # Isolate pixels with detected pothole depression
    pothole_active = depression > 1e-4

    if not np.any(pothole_active):
        return frame.copy()

    # Consistent normalization: use fixed nominal scale (0.8) across all frames
    # so heatmap colors maintain stable, consistent meaning without per-frame flickering.
    NORM_SCALE = 0.8
    normalized = np.clip(
        depression / NORM_SCALE,
        0.0,
        1.0
    )

    heatmap = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )

    # Transparency: non-pothole road pixels have opacity 0.0 (100% transparent).
    # Only pixels inside YOLO pothole segmentation regions get blended.
    opacity = np.zeros((*depression.shape[:2], 1), dtype=np.float32)
    opacity[pothole_active] = np.clip(
        np.sqrt(normalized[pothole_active]) * alpha,
        0.15,
        alpha
    )[:, None]

    return (
        frame * (1.0 - opacity)
        + heatmap * opacity
    ).astype(np.uint8)


# --------------------------------------------------
# SELECT DISPLAY MODE
# --------------------------------------------------

def render_mode(
    frame,
    result,
    mode,
    depression=None,
    alpha=0.85
):

    if mode == "box":
        return render_boxes(frame, result)

    if mode == "seg":
        return render_segmentation(frame, result)

    if mode == "depth":
        return depth_overlay(
            frame,
            depression,
            alpha
        )

    return frame.copy()


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--ckpt",
        default=str(
            HERE
            / "models"
            / "depth_anything_v2_vits.pth"
        ),
        help="Path to Depth Anything V2 weights (default: models/depth_anything_v2_vits.pth)"
    )

    parser.add_argument(
        "--depth-size",
        type=int,
        default=518,
        help="DA-V2 input processing resolution (default: 518; try 392 or 256 for faster CPU)"
    )

    parser.add_argument(
        "--mode",
        choices=["box", "seg", "depth"],
        default="depth",
        help="Initial display mode: 'depth' (heatmap), 'box' (bounding boxes), or 'seg' (segmentation)"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after processing N frames (0 = run continuously until end of stream)"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless execution mode without cv2.imshow GUI window (useful for background processing)"
    )

    parser.add_argument(
        "--no-depth",
        action="store_true"
    )

    parser.add_argument(
        "--depth-alpha",
        type=float,
        default=0.85
    )

    parser.add_argument(
        "--depth-interval",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--no-pace",
        action="store_true"
    )

    parser.add_argument(
        "--out",
        default="output"
    )

    parser.add_argument(
        "--save-video",
        action="store_true"
    )

    parser.add_argument(
        "--max-track-records",
        type=int,
        default=6000
    )

    args = parser.parse_args()

    # Create output folders.

    output_dir = Path(args.out)

    tracking_dir = (
        output_dir / "scene_tracking"
    )

    tracking_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load fine-tuned YOLO model.

    pd.IMAGE_SIZE = args.imgsz

    model, device = pd.load_model()

    print("YOLO detection model ready.")

    # Select PyTorch device for geometry model.

    dev = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Start depression model worker.

    depth = None

    if not args.no_depth and Path(args.ckpt).exists():

        try:

            depth = DepthWorker(
                args.ckpt,
                dev,
                size=args.depth_size,
                interval=args.depth_interval
            )

            depth.start()

            print(f"Depth Anything V2 (ViT-S) loaded from: {args.ckpt}")

        except Exception as error:

            print(
                f"Could not load Depth Anything V2 model: {error}"
            )

            depth = None

    elif not args.no_depth:

        print(
            f"DA-V2 checkpoint missing: {args.ckpt}"
        )

        print("Depression heatmap disabled.")

    # Start video source.

    grabber = Grabber(
        args.source,
        not args.no_pace
    )

    grabber.start()

    # Tracking state.

    prev_gray = None
    points = []
    next_id = 0
    last_id = 0

    records = []

    show_tracks = False

    alpha = args.depth_alpha

    display_mode = args.mode

    writer = None

    fps = 0.0
    previous_time = time.time()

    frame_index = 0
    frame_size = None

    # Keep the latest completed depth result paired with its source frame.
    last_depth_map = None
    last_depth_frame = None
    last_depth_fidx = -1

    # Open display window.

    if not args.no_display:
        cv2.namedWindow(
            "PothVis3D Live",
            cv2.WINDOW_NORMAL
        )

        cv2.createTrackbar(
            "Depth opacity %",
            "PothVis3D Live",
            int(args.depth_alpha * 100),
            100,
            lambda value: None
        )

    print("\nPothVis3D started.")
    print("1 = Bounding boxes")
    print("2 = Segmentation")
    print("3 = Depression heatmap")
    print("t = Toggle tracking")
    print("d = Toggle depression worker")
    print("s = Save snapshot")
    print("q = Quit\n")

    # --------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------

    try:

        while True:

            frame, new_id = grabber.latest(last_id)

            if frame is None:

                if not grabber.ok:
                    break

                time.sleep(0.002)
                continue

            last_id = new_id

            original_height, original_width = frame.shape[:2]

            if original_width != args.width:

                frame = cv2.resize(
                    frame,
                    (
                        args.width,
                        round(
                            original_height
                            * args.width
                            / original_width
                        )
                    )
                )

            frame_size = frame.shape[:2]

            start_time = time.time()

            # ------------------------------------------
            # YOLO DETECTION AND SEGMENTATION
            # ------------------------------------------

            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=pd.CONFIDENCE,
                iou=pd.IOU_THRESHOLD,
                max_det=pd.MAX_DETECTIONS,
                device=device,
                half=(device != "cpu"),
                verbose=False
            )[0]

            if not args.no_display:
                alpha = (
                    cv2.getTrackbarPos(
                        "Depth opacity %",
                        "PothVis3D Live"
                    ) / 100
                )
            else:
                alpha = args.depth_alpha

            # Internal mask used by tracking.

            pothole_mask, pothole_count = build_pothole_mask(
                result,
                frame.shape
            )

            # ------------------------------------------
            # DEPRESSION MAP
            # ------------------------------------------

            if depth is not None and depth.enabled:
                depth.submit(frame, pothole_mask, frame_index)

            # Read the latest completed map and its matching source frame.
            if depth is not None:
                new_depth_map, new_depth_frame, new_depth_fidx = (
                    depth.get_latest()
                )

                if new_depth_map is not None and new_depth_fidx != last_depth_fidx:
                    last_depth_map = new_depth_map
                    last_depth_frame = new_depth_frame
                    last_depth_fidx = new_depth_fidx

            # ------------------------------------------
            # DISPLAY SELECTED MODE
            # ------------------------------------------

            if display_mode == "depth":
                if last_depth_map is not None and last_depth_frame is not None:
                    canvas = depth_overlay(
                        last_depth_frame,
                        last_depth_map,
                        alpha
                    )
                else:
                    canvas = frame.copy()
                    cv2.putText(
                        canvas,
                        "Waiting for first depth result...",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
            else:
                canvas = render_mode(
                    frame,
                    result,
                    display_mode,
                    None,
                    alpha
                )

            detection_time = (
                time.time() - start_time
            ) * 1000

            # ------------------------------------------
            # CAMERA TRACKING
            # ------------------------------------------

            tracking_start = time.time()

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            motion = {
                "valid": False,
                "model": None,
                "inliers": 0,
                "matrix": None
            }

            if (
                prev_gray is None
                or prev_gray.shape != gray.shape
            ):

                points, next_id = st.detect_new_points(
                    gray,
                    pothole_mask,
                    [],
                    next_id
                )

            else:

                points, correspondences = st.track_points(
                    prev_gray,
                    gray,
                    points
                )

                old_points, new_points = (
                    correspondences
                    if len(correspondences) == 2
                    else ([], [])
                )

                motion = st.estimate_camera_motion(
                    old_points,
                    new_points
                )

                points, next_id = st.detect_new_points(
                    gray,
                    pothole_mask,
                    points,
                    next_id
                )

            prev_gray = gray

            if show_tracks:

                canvas = st.draw_tracks(
                    canvas,
                    points,
                    frame_index,
                    motion
                )

            # ------------------------------------------
            # SAVE TRACK RECORDS
            # ------------------------------------------

            records.append({

                "frame": f"live_{frame_index:06d}",

                "frame_index": frame_index,

                "tracks": [
                    {
                        "id": point["id"],

                        "x": round(
                            float(point["point"][0]),
                            2
                        ),

                        "y": round(
                            float(point["point"][1]),
                            2
                        ),

                        "category": point["category"]
                    }

                    for point in points
                ]

            })

            if len(records) > args.max_track_records:
                records.pop(0)

            tracking_time = (
                time.time() - tracking_start
            ) * 1000

            # ------------------------------------------
            # DISPLAY INFORMATION
            # ------------------------------------------

            now = time.time()

            fps = (
                0.9 * fps
                + 0.1 / max(now - previous_time, 1e-6)
            )

            previous_time = now

            mode_names = {
                "box": "BOUNDING BOX",
                "seg": "SEGMENTATION",
                "depth": "DEPRESSION HEATMAP (DA-V2)"
            }

            mode_name = mode_names[display_mode]

            hud = (
                f"{mode_name} | "
                f"FPS {fps:.1f} | "
                f"Potholes {pothole_count}"
            )

            cv2.rectangle(
                canvas,
                (0, 0),
                (canvas.shape[1], 25),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                canvas,
                hud,
                (7, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            if not args.no_display:
                cv2.imshow(
                    "PothVis3D Live",
                    canvas
                )

            # ------------------------------------------
            # SAVE OUTPUT VIDEO
            # ------------------------------------------

            if args.save_video:

                if writer is None:

                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    writer = cv2.VideoWriter(
                        str(
                            output_dir / "live_result.mp4"
                        ),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        18,
                        (
                            canvas.shape[1],
                            canvas.shape[0]
                        )
                    )

                writer.write(canvas)

            frame_index += 1

            if args.max_frames > 0 and frame_index >= args.max_frames:
                print(f"Reached limit of {args.max_frames} frames. Stopping.")
                break

            # ------------------------------------------
            # KEYBOARD CONTROLS
            # ------------------------------------------

            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("1"):
                    display_mode = "box"

                elif key == ord("2"):
                    display_mode = "seg"

                elif key == ord("3"):
                    display_mode = "depth"

                elif key == ord("t"):
                    show_tracks = not show_tracks

                elif key == ord("d") and depth is not None:
                    depth.enabled = not depth.enabled

                elif key == ord("]"):

                    cv2.setTrackbarPos(
                        "Depth opacity %",
                        "PothVis3D Live",
                        min(100, int(alpha * 100) + 10)
                    )

                elif key == ord("["):

                    cv2.setTrackbarPos(
                        "Depth opacity %",
                        "PothVis3D Live",
                        max(0, int(alpha * 100) - 10)
                    )

                elif key == ord("s"):

                    snapshot_path = (
                        output_dir
                        / f"snap_{frame_index:06d}_{display_mode}.png"
                    )

                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    cv2.imwrite(
                        str(snapshot_path),
                        canvas
                    )

                    print(f"Snapshot saved: {snapshot_path}")

    finally:

        # ----------------------------------------------
        # CLEANUP
        # ----------------------------------------------

        grabber.release()

        if depth is not None:
            depth.stop()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    # ----------------------------------------------
    # OPTIONAL CAMERA RECONSTRUCTION
    # ----------------------------------------------

    should_reconstruct = False
    try:
        if (
            len(records) > 30
            and frame_size is not None
            and sys.stdin.isatty()
        ):
            should_reconstruct = (
                input("\nGenerate camera poses and sparse points? [y/N] ").strip().lower() == "y"
            )
    except (EOFError, KeyboardInterrupt):
        should_reconstruct = False

    if should_reconstruct:

        tracks_path = (
            tracking_dir / "tracks.json"
        )

        with open(
            tracks_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                records,
                file,
                indent=2
            )

        import camera_pose_generator as cpg

        height, width = frame_size

        sys.argv = [
            "camera_pose_generator",

            "--tracks",
            str(tracks_path),

            "--out",
            str(output_dir / "reconstruction"),

            "--width",
            str(width),

            "--height",
            str(height)
        ]

        cpg.main()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
