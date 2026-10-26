"""Datasets + preprocessing shared by training and inference."""
from __future__ import annotations
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from synthetic import render_sample

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 RGB [H,W,3] -> normalized float [3,H,W]."""
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


def denorm(t: torch.Tensor) -> np.ndarray:
    """normalized [3,H,W] -> uint8 RGB [H,W,3]."""
    return (np.clip(t.cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1) * 255).astype(np.uint8)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Aspect-preserving resize + zero pad; returns image and scale/pad metadata."""
    h, w = img.shape[:2]; s = size / max(h, w); nh, nw = round(h * s), round(w * s)
    r = img if (nh, nw) == (h, w) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), img.dtype); py, px = (size - nh) // 2, (size - nw) // 2
    out[py:py + nh, px:px + nw] = r
    return out, {"scale": s, "pad_x": px, "pad_y": py, "orig_hw": [h, w]}


class SyntheticPotholes(Dataset):
    """On-the-fly synthetic samples; sample i uses seed (seed0 + offset + i)."""
    def __init__(self, n: int, seed0: int, size: int = 512, dep_scale: float = 0.1):
        self.n, self.seed0, self.size, self.dep_scale, self.offset = n, seed0, size, dep_scale, 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s = render_sample(self.seed0 + self.offset + i, self.size)
        assert np.isfinite(s["depression"]).all()
        return {"image": to_tensor(s["image"]), "mask": torch.from_numpy(s["mask"])[None],
                "depression": torch.from_numpy(s["depression"] / self.dep_scale)[None], "valid": torch.from_numpy(s["valid"])[None]}
