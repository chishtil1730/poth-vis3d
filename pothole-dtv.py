import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

DEPTH_DIR = ROOT / "output" / "depth_results"
FRAME_DIRS = [
    ROOT / "output" / "frames",
    ROOT / "images",
]
MODEL_PATH = ROOT / "sam_b.pt"
from ultralytics import YOLO, SAM

DETECTOR_PATH = ROOT / "best.pt"
SAM_PATH = ROOT / "sam_b.pt"

detector = YOLO(str(DETECTOR_PATH))
sam_model = SAM(str(SAM_PATH))

OUTPUT_DIR = ROOT / "output" / "pothole_depth_video"
OUTPUT_VIDEO = OUTPUT_DIR / "pothole_depth_combined.mp4"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_frames"

FPS = 30
CONFIDENCE = 0.25
MAX_FRAMES = 0  # 0 means process all available frames
COLORMAP = cv2.COLORMAP_TURBO


def get_frame_number(path):
    match = re.search(r"frame_(\d+)", path.stem)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def find_original_frames():
    for directory in FRAME_DIRS:
        if not directory.exists():
            continue

        files = [
            p for p in directory.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        if files:
            return {
                get_frame_number(p): p
                for p in files
                if get_frame_number(p) >= 0
            }

    return {}


def normalize_depth(depth, low, high):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)

    normalized = np.zeros(depth.shape, dtype=np.uint8)

    if not valid.any():
        return normalized

    if high <= low:
        return normalized

    scaled = np.clip((depth - low) / (high - low), 0, 1)
    normalized[valid] = (scaled[valid] * 255).astype(np.uint8)

    return normalized


def prepare_depth(depth):
    depth = np.squeeze(depth)

    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got {depth.shape}")

    valid = np.isfinite(depth)

    if not valid.any():
        return np.zeros(depth.shape, dtype=np.uint8)

    # Use robust percentiles to reduce the influence of outliers.
    low, high = np.percentile(depth[valid], [2, 98])

    return normalize_depth(depth, low, high)


def fit_height(image, target_height):
    height, width = image.shape[:2]

    if height == target_height:
        return image

    new_width = max(1, round(width * target_height / height))

    return cv2.resize(
        image,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA if height > target_height else cv2.INTER_LINEAR,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    if not DEPTH_DIR.exists():
        raise FileNotFoundError(f"Depth directory not found: {DEPTH_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    depth_files = sorted(
        DEPTH_DIR.glob("frame_*_depression.npy"),
        key=get_frame_number,
    )

    if not depth_files:
        raise FileNotFoundError(
            f"No frame_*_depression.npy files found in {DEPTH_DIR}"
        )

    original_frames = find_original_frames()

    if not original_frames:
        raise FileNotFoundError(
            "No original frames found. Checked output/frames and images."
        )

    model = YOLO(str(MODEL_PATH))

    # Use a fixed color scale for every frame so the video is more comparable.
    all_min = float("inf")
    all_max = float("-inf")

    for path in depth_files:
        depth = np.squeeze(np.load(path, mmap_mode="r"))

        if depth.ndim != 2:
            continue

        valid = np.isfinite(depth)

        if valid.any():
            all_min = min(all_min, float(np.percentile(depth[valid], 2)))
            all_max = max(all_max, float(np.percentile(depth[valid], 98)))

    if not np.isfinite(all_min) or not np.isfinite(all_max):
        raise RuntimeError("No valid numeric values found in the depression maps.")

    # Match depth maps to original frames using the frame number.
    available = [
        (path, get_frame_number(path))
        for path in depth_files
        if get_frame_number(path) in original_frames
    ]

    if not available:
        raise RuntimeError(
            "Could not match depression maps to images. "
            "Check that their frame numbers correspond."
        )

    if MAX_FRAMES > 0:
        available = available[:MAX_FRAMES]

    first_depth = np.squeeze(np.load(available[0][0]))
    first_image = cv2.imread(str(original_frames[available[0][1]]))

    if first_image is None:
        raise RuntimeError("Could not read the first original image.")

    target_height = 512
    first_image = fit_height(first_image, target_height)

    first_depth_u8 = normalize_depth(first_depth, all_min, all_max)
    first_depth_color = cv2.applyColorMap(first_depth_u8, COLORMAP)
    first_depth_color = fit_height(first_depth_color, target_height)

    panel_width = max(first_image.shape[1], first_depth_color.shape[1])

    first_image = cv2.resize(first_image, (panel_width, target_height))
    first_depth_color = cv2.resize(
        first_depth_color, (panel_width, target_height)
    )

    output_size = (panel_width * 2, target_height)

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        output_size,
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create the output video.")

    total_detections = 0

    try:
        for index, (depth_path, frame_id) in enumerate(available, start=1):
            image_path = original_frames[frame_id]
            original = cv2.imread(str(image_path))

            if original is None:
                print(f"Skipping unreadable image: {image_path.name}")
                continue

            depth = np.squeeze(np.load(depth_path))

            if depth.ndim != 2:
                print(f"Skipping invalid depth map: {depth_path.name}")
                continue

            # Detect potholes on the original frame.
            results = model.predict(
                source=original,
                conf=CONFIDENCE,
                verbose=False,
            )

            annotated = original.copy()
            frame_detections = 0

            for result in results:
                if result.boxes is None:
                    continue

                masks = result.masks.data.cpu().numpy() if result.masks is not None else None

                for det_index, box in enumerate(result.boxes):
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names.get(class_id, str(class_id))

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].cpu().numpy()
                    )

                    if masks is not None and det_index < len(masks):
                        mask = cv2.resize(
                            masks[det_index],
                            (annotated.shape[1], annotated.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ) > 0.5

                        overlay = annotated.copy()
                        overlay[mask] = (0, 0, 255)

                        annotated = cv2.addWeighted(
                            overlay, 0.35, annotated, 0.65, 0
                        )

                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        cv2.drawContours(
                            annotated, contours, -1, (0, 0, 255), 2
                        )

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    label = f"{class_name} {confidence:.2f}"

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    frame_detections += 1

            total_detections += frame_detections

            # Colorize the relative depression map.
            depth_u8 = normalize_depth(depth, all_min, all_max)
            depth_color = cv2.applyColorMap(depth_u8, COLORMAP)

            depth_color = cv2.resize(
                depth_color,
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            annotated = fit_height(annotated, target_height)
            depth_color = fit_height(depth_color, target_height)

            # Label the panels.
            cv2.putText(
                annotated,
                f"Original + Pothole Detection | Frame {frame_id}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                depth_color,
                "Relative Depression Map",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            combined = np.hstack([annotated, depth_color])

            writer.write(combined)

            output_frame_path = ANNOTATED_DIR / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(output_frame_path), annotated)

            if index % 25 == 0 or index == len(available):
                print(
                    f"Processed {index}/{len(available)} frames | "
                    f"Detections in current frame: {frame_detections}"
                )

    finally:
        writer.release()

    print("\nProcessing complete.")
    print(f"Frames processed: {len(available)}")
    print(f"Total detections across frames: {total_detections}")
    print(f"Video saved to: {OUTPUT_VIDEO}")
    print(f"Annotated frames saved to: {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()
