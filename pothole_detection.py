
import cv2
import json
import torch
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "best.pt"

INPUT_DIR = PROJECT_DIR / "output" / "frames"
OUTPUT_DIR = PROJECT_DIR / "output" / "pothole_results"

ANNOTATED_DIR = OUTPUT_DIR / "annotated"
MASKS_DIR = OUTPUT_DIR / "masks"


# --------------------------------------------------
# DETECTION SETTINGS
# --------------------------------------------------

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 30


# --------------------------------------------------
# LOAD FINE-TUNED YOLO MODEL
# --------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}\n"
            "Make sure best.pt is inside the models folder."
        )

    model = YOLO(str(MODEL_PATH))

    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("\nYOLO MODEL LOADED SUCCESSFULLY")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {'CUDA GPU' if device == 0 else 'CPU'}")

    return model, device


# --------------------------------------------------
# DETECT POTHOLES IN AN IMAGE
# --------------------------------------------------

def detect_potholes(model, device, image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return None

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        device=device,
        half=(device != "cpu"),
        verbose=False
    )

    return image, results[0]


# --------------------------------------------------
# PROCESS EXTRACTED VIDEO FRAMES
# --------------------------------------------------

def process_frames():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in {INPUT_DIR}"
        )

    model, device = load_model()

    all_detections = []

    for frame_index, image_path in enumerate(image_paths):

        output = detect_potholes(
            model,
            device,
            image_path
        )

        if output is None:
            continue

        original_image, result = output

        annotated_image = result.plot()

        annotated_path = ANNOTATED_DIR / image_path.name

        cv2.imwrite(
            str(annotated_path),
            annotated_image
        )

        frame_data = {
            "frame": image_path.name,
            "frame_index": frame_index,
            "image_width": int(original_image.shape[1]),
            "image_height": int(original_image.shape[0]),
            "pothole_count": 0,
            "detections": []
        }

        if result.boxes is not None and len(result.boxes):

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
            else:
                masks = []

            for detection_index, (
                box,
                confidence,
                class_id
            ) in enumerate(zip(boxes, confidences, classes)):

                x1, y1, x2, y2 = box.tolist()

                detection = {
                    "detection_id": detection_index,
                    "class_id": int(class_id),
                    "class_name": model.names[int(class_id)],
                    "confidence": round(float(confidence), 4),
                    "bbox_xyxy": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(x2), 2),
                        round(float(y2), 2)
                    ],
                    "mask_file": None
                }

                if detection_index < len(masks):

                    mask = masks[detection_index]

                    mask = cv2.resize(
                        mask,
                        (
                            original_image.shape[1],
                            original_image.shape[0]
                        ),
                        interpolation=cv2.INTER_NEAREST
                    )

                    binary_mask = (
                        mask > 0.5
                    ).astype(np.uint8) * 255

                    mask_name = (
                        f"{image_path.stem}_"
                        f"pothole_{detection_index:03d}.png"
                    )

                    mask_path = MASKS_DIR / mask_name

                    cv2.imwrite(
                        str(mask_path),
                        binary_mask
                    )

                    detection["mask_file"] = str(mask_path)

                frame_data["detections"].append(detection)

        frame_data["pothole_count"] = len(
            frame_data["detections"]
        )

        all_detections.append(frame_data)

        print(
            f"[{frame_index + 1}/{len(image_paths)}] "
            f"{image_path.name}: "
            f"{frame_data['pothole_count']} pothole(s)"
        )

    # Save detection results as JSON.

    json_path = OUTPUT_DIR / "detections.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_detections, file, indent=2)

    total_detections = sum(
        frame["pothole_count"]
        for frame in all_detections
    )

    print("\nPROCESSING COMPLETE")
    print(f"Frames processed: {len(all_detections)}")
    print(f"Total frame-level detections: {total_detections}")
    print(f"Annotated images: {ANNOTATED_DIR}")
    print(f"Pothole masks: {MASKS_DIR}")
    print(f"Detection data: {json_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    process_frames()
