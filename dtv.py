import re
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path("output/depth_results")
OUTPUT_VIDEO = Path("output/depression_map_video.mp4")

FPS = 30
COLORMAP = cv2.COLORMAP_TURBO

def frame_number(path):
    match = re.search(r"frame_(\d+)_depression\.npy$", path.name)
    return int(match.group(1)) if match else -1

def normalize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)

    if not valid.any():
        return np.zeros(depth.shape, dtype=np.uint8)

    low, high = np.percentile(depth[valid], [2, 98])

    if high <= low:
        return np.zeros(depth.shape, dtype=np.uint8)

    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (depth[valid] - low) / (high - low), 0, 1
    )

    return (normalized * 255).astype(np.uint8)

def main():
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        INPUT_DIR.glob("frame_*_depression.npy"),
        key=frame_number
    )

    if not files:
        raise FileNotFoundError(
            f"No depression maps found in {INPUT_DIR.resolve()}"
        )

    first = np.load(files[0])
    if first.ndim != 2:
        first = np.squeeze(first)

    height, width = first.shape

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create the output video.")

    try:
        for index, path in enumerate(files, start=1):
            depth = np.load(path)
            depth = np.squeeze(depth)

            if depth.ndim != 2:
                print(f"Skipping unsupported shape: {path.name}, {depth.shape}")
                continue

            if depth.shape != (height, width):
                depth = cv2.resize(
                    depth, (width, height),
                    interpolation=cv2.INTER_LINEAR
                )

            normalized = normalize_depth(depth)
            colored = cv2.applyColorMap(normalized, COLORMAP)

            writer.write(colored)

            if index % 50 == 0 or index == len(files):
                print(f"Processed {index}/{len(files)} maps")

    finally:
        writer.release()

    print(f"\nVideo saved to: {OUTPUT_VIDEO.resolve()}")
    print(f"Input maps: {len(files)}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {FPS}")
    print(f"Approximate duration: {len(files) / FPS:.2f} seconds")

if __name__ == "__main__":
    main()
