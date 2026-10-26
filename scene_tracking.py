import cv2
import json
import numpy as np

from pathlib import Path
from collections import deque

INPUT_DIR = Path("output/frames")
DETECTIONS_FILE = Path("output/pothole_results/detections.json")
OUTPUT_DIR = Path("output/scene_tracking")
TRACKED_DIR = OUTPUT_DIR / "tracked_frames"

MAX_POINTS = 300
MAX_TRAIL_LENGTH = 20
MIN_DISTANCE = 8
QUALITY_LEVEL = 0.01
MIN_TRACKS_FOR_MOTION = 8
FB_THRESHOLD = 1.5

FEATURE_PARAMS = {
    "maxCorners": MAX_POINTS,
    "qualityLevel": QUALITY_LEVEL,
    "minDistance": MIN_DISTANCE,
    "blockSize": 7
}

LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
}


def load_detections():
    if not DETECTIONS_FILE.exists():
        print("Detection JSON not found. Tracking road features only.")
        return {}

    with open(DETECTIONS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {item["frame"]: item for item in data}


def get_pothole_mask(frame_name, frame_shape, detections):
    height, width = frame_shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    frame_data = detections.get(frame_name)

    if frame_data is None:
        return combined_mask

    for detection in frame_data.get("detections", []):
        mask_file = detection.get("mask_file")

        if not mask_file:
            continue

        mask_path = Path(mask_file)

        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        combined_mask = cv2.bitwise_or(
            combined_mask,
            (mask > 127).astype(np.uint8) * 255
        )

    return combined_mask


def detect_new_points(gray, pothole_mask, existing_points, next_id):
    height, width = gray.shape

    if existing_points:
        points = np.array(
            [item["point"] for item in existing_points],
            dtype=np.float32
        ).reshape(-1, 1, 2)
    else:
        points = np.empty((0, 1, 2), dtype=np.float32)

    categories = ["pothole", "road"]

    for category in categories:
        category_mask = np.zeros((height, width), dtype=np.uint8)

        if category == "pothole":
            category_mask = pothole_mask.copy()
        else:
            category_mask = cv2.bitwise_not(pothole_mask)

        # Avoid rediscovering points close to existing tracks.
        for point in points.reshape(-1, 2):
            x, y = np.round(point).astype(int)

            cv2.circle(
                category_mask,
                (x, y),
                MIN_DISTANCE,
                0,
                -1
            )

        current_category_count = sum(
            item["category"] == category
            for item in existing_points
        )

        target_count = (
            MAX_POINTS // 3
            if category == "pothole"
            else MAX_POINTS
        )

        available = target_count - current_category_count

        if available <= 0:
            continue

        params = FEATURE_PARAMS.copy()
        params["maxCorners"] = available

        corners = cv2.goodFeaturesToTrack(
            gray,
            mask=category_mask,
            **params
        )

        if corners is None:
            continue

        for corner in corners.reshape(-1, 2):
            if len(existing_points) >= MAX_POINTS:
                break

            x, y = corner.tolist()

            existing_points.append({
                "id": next_id,
                "point": [float(x), float(y)],
                "category": category,
                "trail": deque(
                    [(float(x), float(y))],
                    maxlen=MAX_TRAIL_LENGTH
                )
            })

            next_id += 1

    return existing_points, next_id


def track_points(previous_gray, current_gray, points):
    if not points:
        return [], []

    previous_points = np.array(
        [item["point"] for item in points],
        dtype=np.float32
    ).reshape(-1, 1, 2)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        **LK_PARAMS
    )

    if next_points is None or status is None:
        return [], []

    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        next_points,
        None,
        **LK_PARAMS
    )

    if backward_points is None or backward_status is None:
        return [], []

    height, width = current_gray.shape

    forward_backward_error = np.linalg.norm(
        previous_points.reshape(-1, 2) -
        backward_points.reshape(-1, 2),
        axis=1
    )

    tracked = []
    old_positions = []
    new_positions = []

    for index, item in enumerate(points):
        x, y = next_points[index, 0]

        valid = (
            status[index, 0] == 1
            and backward_status[index, 0] == 1
            and forward_backward_error[index] < FB_THRESHOLD
            and 0 <= x < width
            and 0 <= y < height
        )

        if not valid:
            continue

        updated = {
            "id": item["id"],
            "point": [float(x), float(y)],
            "category": item["category"],
            "trail": deque(
                item["trail"],
                maxlen=MAX_TRAIL_LENGTH
            )
        }

        updated["trail"].append((float(x), float(y)))

        tracked.append(updated)

        old_positions.append(
            previous_points[index, 0].copy()
        )

        new_positions.append(
            np.array([x, y], dtype=np.float32)
        )

    return tracked, (
        np.array(old_positions, dtype=np.float32),
        np.array(new_positions, dtype=np.float32)
    )


def estimate_camera_motion(old_points, new_points):
    if len(old_points) < MIN_TRACKS_FOR_MOTION:
        return {
            "valid": False,
            "model": None,
            "inliers": 0,
            "matrix": None
        }

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=1000,
        confidence=0.99
    )

    if matrix is None or inlier_mask is None:
        return {
            "valid": False,
            "model": None,
            "inliers": 0,
            "matrix": None
        }

    inliers = int(inlier_mask.sum())

    if inliers < MIN_TRACKS_FOR_MOTION:
        return {
            "valid": False,
            "model": None,
            "inliers": inliers,
            "matrix": matrix.tolist()
        }

    return {
        "valid": True,
        "model": "partial_affine_ransac",
        "inliers": inliers,
        "matrix": matrix.tolist()
    }


def draw_tracks(frame, points, frame_index, motion):
    output = frame.copy()

    for item in points:
        x, y = np.round(item["point"]).astype(int)

        if item["category"] == "pothole":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        trail = list(item["trail"])

        for i in range(1, len(trail)):
            p1 = tuple(np.round(trail[i - 1]).astype(int))
            p2 = tuple(np.round(trail[i]).astype(int))

            cv2.line(output, p1, p2, color, 1)

        cv2.circle(output, (x, y), 3, color, -1)

        cv2.putText(
            output,
            str(item["id"]),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

    motion_text = (
        f"Motion inliers: {motion['inliers']}"
    )

    cv2.putText(
        output,
        f"Frame: {frame_index} | Tracks: {len(points)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        motion_text,
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return output


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input frames not found: {INPUT_DIR}"
        )

    TRACKED_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    frame_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    )

    if not frame_paths:
        raise RuntimeError("No input frames found.")

    detections = load_detections()

    first_frame = cv2.imread(str(frame_paths[0]))

    if first_frame is None:
        raise RuntimeError("Could not read the first frame.")

    previous_gray = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2GRAY
    )

    previous_mask = get_pothole_mask(
        frame_paths[0].name,
        first_frame.shape,
        detections
    )

    points = []
    next_id = 0

    points, next_id = detect_new_points(
        previous_gray,
        previous_mask,
        points,
        next_id
    )

    all_frames_data = []
    previous_frame_index = 0

    for frame_index, frame_path in enumerate(frame_paths):
        if frame_index == 0:
            frame = first_frame.copy()
            motion = {
                "valid": False,
                "model": None,
                "inliers": 0,
                "matrix": None
            }
        else:
            frame = cv2.imread(str(frame_path))

            if frame is None:
                print(f"Skipping unreadable frame: {frame_path}")
                continue

            current_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            current_mask = get_pothole_mask(
                frame_path.name,
                frame.shape,
                detections
            )

            points, correspondences = track_points(
                previous_gray,
                current_gray,
                points
            )

            old_points, new_points = correspondences

            motion = estimate_camera_motion(
                old_points,
                new_points
            )

            points, next_id = detect_new_points(
                current_gray,
                current_mask,
                points,
                next_id
            )

            previous_gray = current_gray
            previous_mask = current_mask
            previous_frame_index = frame_index

        annotated = draw_tracks(
            frame,
            points,
            frame_index,
            motion
        )

        output_path = TRACKED_DIR / frame_path.name

        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(
                f"Could not write frame: {output_path}"
            )

        frame_data = {
            "frame": frame_path.name,
            "frame_index": frame_index,
            "tracks": [
                {
                    "id": item["id"],
                    "x": round(item["point"][0], 3),
                    "y": round(item["point"][1], 3),
                    "category": item["category"]
                }
                for item in points
            ],
            "camera_motion": motion
        }

        all_frames_data.append(frame_data)

        print(
            f"[{frame_index + 1}/{len(frame_paths)}] "
            f"{frame_path.name}: "
            f"{len(points)} active points"
        )

    json_path = OUTPUT_DIR / "tracks.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_frames_data, file, indent=2)

    print("\nTracking completed.")
    print(f"Tracked frames: {TRACKED_DIR}")
    print(f"Tracking data: {json_path}")


if __name__ == "__main__":
    main()
