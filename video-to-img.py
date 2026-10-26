import cv2
import numpy as np
import argparse
from pathlib import Path


TARGET_WIDTH = 512
TARGET_HEIGHT = 512
TARGET_FPS = 24


def normalize_frame(frame, width=TARGET_WIDTH, height=TARGET_HEIGHT):
    original_height, original_width = frame.shape[:2]

    scale = min(width / original_width, height / original_height)

    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


def process_video(
    input_source,
    output_video=None,
    frames_dir=None,
    target_fps=TARGET_FPS
):
    is_camera = isinstance(input_source, int) or str(input_source).isdigit()
    source = int(input_source) if is_camera else str(input_source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {input_source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = None

    frame_count = 0
    output_count = 0
    next_output_time = 0.0

    writer = None

    if frames_dir:
        frames_path = Path(frames_dir)
        frames_path.mkdir(parents=True, exist_ok=True)
    else:
        frames_path = None

    if output_video:
        output_path = Path(output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            target_fps,
            (TARGET_WIDTH, TARGET_HEIGHT)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {output_video}")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            if source_fps is None:
                # Without reliable source timestamps, process every frame.
                # This is suitable for live cameras, but cannot guarantee
                # accurate temporal resampling of an unknown-rate video.
                should_process = True
            else:
                frame_time = frame_count / source_fps
                should_process = frame_time + 1e-9 >= next_output_time

            if should_process:
                normalized = normalize_frame(frame)

                if writer is not None:
                    writer.write(normalized)

                if frames_path is not None:
                    frame_path = frames_path / f"frame_{output_count:06d}.jpg"
                    if not cv2.imwrite(str(frame_path), normalized):
                        raise RuntimeError(f"Could not save frame: {frame_path}")

                output_count += 1

                if source_fps is not None:
                    next_output_time = output_count / target_fps

            frame_count += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Frames read: {frame_count}")
    print(f"Frames processed: {output_count}")
    print(f"Output resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Target FPS: {target_fps}")

    if output_video:
        print(f"Normalized video: {output_video}")

    if frames_path is not None:
        print(f"Extracted frames: {frames_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize video to 512x512 at 24 FPS."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path or camera index, such as 0."
    )

    parser.add_argument(
        "--output",
        default="output/normalized_video.mp4",
        help="Output normalized video path."
    )

    parser.add_argument(
        "--frames",
        default=None,
        help="Optional directory for saving normalized frames."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=TARGET_FPS,
        help="Target output FPS (default: 24)."
    )

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero.")

    process_video(
        input_source=args.input,
        output_video=args.output,
        frames_dir=args.frames,
        target_fps=args.fps
    )
