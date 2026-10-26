import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def load_tracks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tracks.json must contain a list of frame records.")
    data.sort(key=lambda item: item.get("frame_index", 0))
    return data


def frame_track_map(frame_record, categories):
    result = {}
    for track in frame_record.get("tracks", []):
        if categories and track.get("category") not in categories:
            continue
        try:
            track_id = int(track["id"])
            x, y = float(track["x"]), float(track["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            result[track_id] = np.array([x, y], dtype=np.float32)
    return result


def correspondences(record_a, record_b, categories):
    a = frame_track_map(record_a, categories)
    b = frame_track_map(record_b, categories)
    shared = sorted(set(a).intersection(b))
    if len(shared) < 8:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, []
    p1 = np.array([a[i] for i in shared], dtype=np.float32)
    p2 = np.array([b[i] for i in shared], dtype=np.float32)
    return p1, p2, shared


def make_intrinsics(width, height, focal_px):
    if focal_px <= 0:
        # A conservative rough guess only. Real calibration is preferable.
        focal_px = 1.2 * max(width, height)
    return np.array([
        [focal_px, 0.0, (width - 1) / 2.0],
        [0.0, focal_px, (height - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def triangulate_points(p1, p2, K, R, t, pose_mask, max_reprojection_error=2.0,
                       min_parallax_deg=1.0, max_depth=1e4):
    if pose_mask is None:
        return np.empty((0, 3), dtype=np.float64)
    mask = pose_mask.reshape(-1) > 0
    if int(mask.sum()) < 8:
        return np.empty((0, 3), dtype=np.float64)

    a, b = p1[mask], p2[mask]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    homog = cv2.triangulatePoints(P1, P2, a.T, b.T)
    w = homog[3]
    valid_w = np.abs(w) > 1e-9
    xyz = np.full((homog.shape[1], 3), np.nan, dtype=np.float64)
    xyz[valid_w] = (homog[:3, valid_w] / w[valid_w]).T

    xyz2 = (R @ xyz.T + t.reshape(3, 1)).T
    z1, z2 = xyz[:, 2], xyz2[:, 2]

    # Reproject into both cameras and reject geometrically inconsistent points.
    def project(points):
        projected = (K @ points.T).T
        den = projected[:, 2]
        out = np.full((len(points), 2), np.nan, dtype=np.float64)
        ok = np.abs(den) > 1e-9
        out[ok] = projected[ok, :2] / den[ok, None]
        return out

    reproj1 = project(xyz)
    reproj2 = project(xyz2)
    err1 = np.linalg.norm(reproj1 - a, axis=1)
    err2 = np.linalg.norm(reproj2 - b, axis=1)

    # Parallax helps remove unstable points created by nearly identical frames.
    ray1 = xyz
    # Express the second camera ray in camera-1 coordinates.
    # The camera-2 center in camera-1 coordinates is -R.T @ t, so
    # the ray from camera 2 toward each point is xyz + R.T @ t.
    ray2 = xyz + (R.T @ t.reshape(3, 1)).T
    n1 = np.linalg.norm(ray1, axis=1)
    n2 = np.linalg.norm(ray2, axis=1)
    denom = np.maximum(n1 * n2, 1e-12)
    cos_angle = np.clip(np.sum(ray1 * ray2, axis=1) / denom, -1.0, 1.0)
    parallax = np.degrees(np.arccos(cos_angle))

    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(err1) & np.isfinite(err2)
        & (z1 > 0) & (z2 > 0)
        & (np.linalg.norm(xyz, axis=1) < max_depth)
        & (err1 <= max_reprojection_error)
        & (err2 <= max_reprojection_error)
        & (parallax >= min_parallax_deg)
    )
    return xyz[valid]


def voxel_downsample(points, voxel_size):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or voxel_size <= 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def robust_filter(points, max_radius_percentile=98.0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 20:
        return points[np.isfinite(points).all(axis=1)]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        return points
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    cutoff = np.percentile(radii, max_radius_percentile)
    return points[radii <= cutoff]


def write_ply(path, points, colors=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is None:
        colors = np.tile(np.array([[100, 200, 255]], dtype=np.uint8), (len(points), 1))
    else:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[valid]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_trajectory_obj(path, points):
    """Write an OBJ polyline, so the trajectory is an actual line, not a point cloud."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("# Relative monocular camera trajectory; scale is arbitrary\n")
        for p in points:
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        if len(points) >= 2:
            f.write("l " + " ".join(str(i + 1) for i in range(len(points))) + "\n")


def save_pose(poses, record, center, rotation, valid):
    poses.append({
        "frame": record.get("frame"),
        "frame_index": record.get("frame_index", 0),
        "valid": bool(valid),
        "camera_center": np.asarray(center, dtype=float).tolist(),
        "rotation_world_camera": np.asarray(rotation, dtype=float).tolist(),
        "translation_scale": "arbitrary_monocular",
    })


def main():
    parser = argparse.ArgumentParser(
        description="Estimate relative monocular camera poses and a filtered sparse point cloud from tracks.json."
    )
    parser.add_argument("--tracks", default="output/scene_tracking/tracks.json")
    parser.add_argument("--out", default="output/reconstruction")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--focal", type=float, default=0.0,
                        help="Approximate focal length in pixels; 0 uses a rough guess.")
    parser.add_argument("--stride", type=int, default=3,
                        help="Compare frames this many records apart. Try 2-5 for more baseline.")
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--categories", default="road",
                        help="Comma-separated track categories, e.g. road or road,pothole.")
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--max-reprojection-error", type=float, default=2.0)
    parser.add_argument("--min-parallax-deg", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=10000.0,
                        help="Reject triangulated points farther than this arbitrary-scale distance.")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                        help="Voxel size for duplicate-point removal in arbitrary reconstruction units; 0 disables.")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")

    tracks_path = Path(args.tracks)
    if not tracks_path.exists():
        raise FileNotFoundError(f"Track file not found: {tracks_path.resolve()}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_tracks(tracks_path)
    if len(records) < 2:
        raise RuntimeError("At least two tracked frames are required.")

    categories = {s.strip() for s in args.categories.split(",") if s.strip()}
    K = make_intrinsics(args.width, args.height, args.focal)

    # R_wc maps camera coordinates into world coordinates; C is camera center in world.
    R_wc = np.eye(3, dtype=np.float64)
    C = np.zeros(3, dtype=np.float64)
    poses = []
    save_pose(poses, records[0], C, R_wc, True)
    valid_centers = [C.copy()]
    all_points = []
    stats = []

    for idx in range(args.stride, len(records), args.stride):
        prev, curr = records[idx - args.stride], records[idx]
        p1, p2, shared_ids = correspondences(prev, curr, categories)
        entry = {
            "from_frame": prev.get("frame"),
            "to_frame": curr.get("frame"),
            "from_frame_index": prev.get("frame_index", idx - args.stride),
            "to_frame_index": curr.get("frame_index", idx),
            "matches": len(shared_ids), "inliers": 0, "valid": False,
            "camera_center": None,
        }

        R_new, C_new = R_wc.copy(), C.copy()
        pose_ok = False
        if len(shared_ids) >= max(8, args.min_inliers):
            E, mask = cv2.findEssentialMat(
                p1, p2, K, method=cv2.RANSAC, prob=0.999,
                threshold=args.ransac_threshold
            )
            if E is not None and mask is not None:
                if E.shape[0] > 3:
                    E = E[:3, :3]
                try:
                    inliers, R_rel, t_rel, pose_mask = cv2.recoverPose(E, p1, p2, K, mask=mask.copy())
                except cv2.error:
                    inliers, R_rel, t_rel, pose_mask = 0, None, None, None
                entry["inliers"] = int(inliers)
                if R_rel is not None and t_rel is not None and inliers >= args.min_inliers:
                    # X_curr = R_rel * X_prev + t_rel
                    R_new = R_wc @ R_rel.T
                    C_new = C - R_new @ t_rel.reshape(3)
                    pose_ok = np.isfinite(C_new).all() and np.isfinite(R_new).all()
                    if pose_ok:
                        new_points = triangulate_points(
                            p1, p2, K, R_rel, t_rel.reshape(3), pose_mask,
                            args.max_reprojection_error, args.min_parallax_deg,
                            args.max_depth
                        )
                        if len(new_points):
                            points_world = (R_wc @ new_points.T).T + C
                            all_points.append(points_world)

        if pose_ok:
            R_wc, C = R_new, C_new
            valid_centers.append(C.copy())
            entry["valid"] = True
            entry["camera_center"] = C.tolist()
        # Keep a record for every input frame; failed estimates do not enter the exported trajectory.
        save_pose(poses, curr, C, R_wc, pose_ok)
        stats.append(entry)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        points = robust_filter(points)
        points = voxel_downsample(points, args.voxel_size)
        if len(points) > args.max_points:
            selection = np.linspace(0, len(points) - 1, args.max_points).astype(int)
            points = points[selection]
    else:
        points = np.empty((0, 3), dtype=np.float64)

    trajectory = np.asarray(valid_centers, dtype=np.float64).reshape(-1, 3)
    write_ply(out_dir / "camera_trajectory.ply", trajectory)
    write_trajectory_obj(out_dir / "camera_trajectory.obj", trajectory)
    write_ply(out_dir / "sparse_points.ply", points)

    payload = {
        "description": (
            "Filtered relative monocular reconstruction. Camera translation scale is arbitrary; "
            "this is not a metric road surface. Invalid pose estimates are excluded from trajectory exports."
        ),
        "intrinsics": K.tolist(),
        "image_size": [args.width, args.height],
        "categories_used": sorted(categories),
        "poses": poses,
        "pair_statistics": stats,
        "summary": {
            "input_frames": len(records),
            "pose_records": len(poses),
            "valid_motion_pairs": sum(s["valid"] for s in stats),
            "attempted_motion_pairs": len(stats),
            "trajectory_vertices": int(len(trajectory)),
            "sparse_points": int(len(points)),
            "translation_scale": "arbitrary_monocular",
            "note": "PLY files contain vertices only. camera_trajectory.obj contains a connected line through valid camera centers."
        },
    }
    with open(out_dir / "camera_poses.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    valid_pairs = sum(s["valid"] for s in stats)
    print(f"Input frames: {len(records)}")
    print(f"Valid motion pairs: {valid_pairs} / {len(stats)}")
    print(f"Valid trajectory vertices: {len(trajectory)}")
    print(f"Filtered sparse 3D points: {len(points)}")
    print(f"Output directory: {out_dir.resolve()}")
    print("Created: camera_poses.json, camera_trajectory.ply, camera_trajectory.obj, sparse_points.ply")
    print("IMPORTANT: Scale is arbitrary. This does not produce metric depth or a road surface mesh.")
    if valid_pairs < max(10, int(0.25 * max(1, len(stats)))):
        print("WARNING: Few motion pairs succeeded. Try --stride 2 or --stride 4 and verify camera intrinsics.")


if __name__ == "__main__":
    main()
