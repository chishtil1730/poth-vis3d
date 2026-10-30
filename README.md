# poth-vis3d

A computer-vision pipeline for pothole detection, relative-depression estimation, scene tracking, and monocular camera-trajectory reconstruction from road video or live cam feed.

## Project overview

- **`video-to-img.py`** — converts an input video into normalized frames in `output/frames/`.
- **`pothole_detection.py`** — detects potholes and saves results under `output/pothole_results/`.
- **`scene_tracking.py`** — tracks features across frames and writes tracking data to `output/scene_tracking/`.
- **`pothole_geometry_net/`** — contains `train.py` for training and `infer.py` for generating pothole probability, relative-depression, and uncertainty maps in `output/depth_results/`.
- **`camera_pose_generator.py`** — estimates a monocular camera trajectory and exports pose and sparse-point-cloud files to `output/reconstruction/`.
- **`dtv.py`** — combines the generated depression maps into a video.
- **`live_pipeline.py`** — intended entry point for processing a camera or video source live, if the script and its dependencies are configured.

> **Note:** The depression output represents relative values, not calibrated metric depth. Monocular reconstruction also has arbitrary scale unless additional calibration or scale information is provided.

## Setting Up

Run commands from the repository root. The examples below use the Windows path from the development environment; change it if your clone cd into that folder

```powershell
cd C:\Users\LENOVO\Dsa\Potholes
conda activate gpu
```

Check the core Python and GPU environment:

```powershell
python -c "import cv2, numpy, torch, yaml, ultralytics, huggingface_hub; print('Core dependencies OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If core packages are missing:

```powershell
pip install opencv-python numpy pyyaml ultralytics huggingface_hub
```

If needed, install the geometry network's requirements:

```powershell
cd pothole_geometry_net
pip install -r requirements.txt
cd ..
```

## Offline processing commands

Run these stages in order when starting from a source video.

### 1. Convert video to frames

```powershell
python video-to-img.py --input "potholes.mp4" --output "output/normalized_video.mp4" --frames "output/frames" --fps 24
```

The frame directory is used as input to detection, tracking, and geometry inference.

### 2. Detect potholes

```powershell
python pothole_detection.py
```

Expected output directory: `output/pothole_results/`.

Run detection before tracking if the tracking script consumes the detection JSON.

### 3. Track scene features

```powershell
python scene_tracking.py
```

Expected output directory: `output/scene_tracking/`, including `tracks.json` and `tracked_frames/`.

### 4. Generate camera poses and sparse reconstruction

```powershell
python camera_pose_generator.py
```

Expected output directory: `output/reconstruction/`, including `camera_poses.json`, `sparse_points.ply`, `camera_trajectory.ply`, and `camera_trajectory.obj`.

Optional explicit parameters, if supported by the current script version:

```powershell
python camera_pose_generator.py --tracks output/scene_tracking/tracks.json --out output/reconstruction --width 512 --height 512 --stride 3 --min-inliers 15
```

### 5. Train the geometry network (optional)

Only run training if you want to train or retrain the model.

```powershell
cd pothole_geometry_net
python train.py --smoke
python train.py --config config.yaml
```

Resume training from a checkpoint:

```powershell
python train.py --config config.yaml --resume outputs/run1/last.pt
```

The configuration & checkpoint paths must match the files actually present in your checkout.

### 6. Run geometry inference

From the `pothole_geometry_net/` directory:

```powershell
python infer.py --ckpt outputs/run1/best.pt --input ../output/frames --out ../output/depth_results --size 512 --batch 4 --device auto --fp16
```

If FP16 causes an error, retry without `--fp16`:

```powershell
python infer.py --ckpt outputs/run1/best.pt --input ../output/frames --out ../output/depth_results --size 512 --batch 4 --device auto
```

Expected output directory: `output/depth_results/`, including depression, probability, uncertainty (`logvar`), and panel outputs.

Return to the repository root after inference:

```powershell
cd ..
```

### 7. Create a depression-map video

```powershell
python dtv.py
```

The script is expected to read depression maps from `output/depth_results/` and write the rendered video to `output/depression_map_video.mp4`.

## Live pipeline commands

These commands assume `live_pipeline.py` exists in the repository root and supports the listed arguments.

### Process a video as fast as possible

```powershell
python live_pipeline.py --source road.mp4 --no-pace
```

### Measure FPS without depth inference

```powershell
python live_pipeline.py --source road.mp4 --no-depth
```

### Run a webcam and save the processed video

```powershell
python live_pipeline.py --source 0 --save-video
```

Expected video output: `output/live_result.mp4`.

### Disable playback pacing and depth inference

```powershell
python live_pipeline.py --source road.mp4 --no-pace --no-depth
```

### Run a webcam without saving a video

```powershell
python live_pipeline.py --source 0
```

### Run an RTSP camera

```powershell
python live_pipeline.py --source "rtsp://camera-address/stream" --save-video
```

Replace the example RTSP URL with the actual camera stream address.

## Suggested end-to-end offline run

Run each command sequentially from the repository root, except for geometry inference, which is run inside `pothole_geometry_net/`.

```powershell
python video-to-img.py --input "potholes.mp4" --output "output/normalized_video.mp4" --frames "output/frames" --fps 24
python pothole_detection.py
python scene_tracking.py
python camera_pose_generator.py
cd pothole_geometry_net
python infer.py --ckpt outputs/run1/best.pt --input ../output/frames --out ../output/depth_results --size 512 --batch 4 --device auto --fp16
cd ..
python dtv.py
```

## Notes

- Replace `potholes.mp4` and `road.mp4` with your actual input filenames.
- Check that the checkpoint path `pothole_geometry_net/outputs/run1/best.pt` exists before inference.
- Run the smoke test before a full training run if you want to verify the training code quickly.
- If a command reports an unrecognized argument, inspect that script's `--help` output. The commands here reflect the project workflow discussed so far and may need adjustment if your local script differs.
- Targeting 16–20 FPS requires benchmarking the complete pipeline on the target camera and hardware; individual model inference FPS does not equal end-to-end FPS.
