import cv2, numpy as np


def colorize(a: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """float map -> BGR turbo image (visualization only)."""
    return cv2.applyColorMap((np.clip((a - vmin) / max(vmax - vmin, 1e-9), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def hstack(tiles: list[np.ndarray]) -> np.ndarray:
    return np.hstack([t if t.ndim == 3 else cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles])
