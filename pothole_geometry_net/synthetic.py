"""Procedural road renderer with exact geometry. Flat-road perspective model:
depression is vertical displacement (metres, synthetic) below the local road plane."""
from __future__ import annotations
import cv2, numpy as np
from scipy.ndimage import map_coordinates


def _noise(rng, X, Z, cells_per_m: float) -> np.ndarray:
    g = rng.standard_normal((128, 128)).astype(np.float32)
    return map_coordinates(g, np.stack([(Z * cells_per_m) % 128, (X * cells_per_m) % 128]), order=1, mode="wrap")


def _ground(size: int, f: float, pitch: float, h: float):
    u, v = np.meshgrid(np.arange(size) + .5, np.arange(size) + .5)
    x, y = (u - size / 2) / f, (v - size / 2) / f
    c, s = np.cos(pitch), np.sin(pitch)
    dy, dz = y * c + s, -y * s + c
    valid = dy > 0.08
    t = np.where(valid, h / np.maximum(dy, 1e-6), 0)
    return (t * x).astype(np.float32), (t * dz).astype(np.float32), valid


class _Pothole:
    def __init__(self, rng, x0: float, z0: float):
        self.x0, self.z0 = x0, z0
        self.a = rng.uniform(.2, .6); self.b = self.a * rng.uniform(.6, 1.4)
        self.rot = rng.uniform(0, np.pi); self.depth = rng.uniform(.03, .12); self.p = rng.uniform(1.5, 4)
        self.harm = [(k, rng.uniform(0, .12) / k * 2, rng.uniform(0, 6.28)) for k in (2, 3, 4, 5)]

    def depth_at(self, X, Z):
        dx, dz = X - self.x0, Z - self.z0
        c, s = np.cos(self.rot), np.sin(self.rot)
        u, w = (c * dx + s * dz) / self.a, (-s * dx + c * dz) / self.b
        th = np.arctan2(w, u)
        rim = 1 + sum(a * np.sin(k * th + ph) for k, a, ph in self.harm)
        r = np.sqrt(u * u + w * w) / np.maximum(rim, .5)
        return self.depth * np.clip(1 - r ** self.p, 0, 1) ** .8


def render_sample(seed: int, size: int = 512) -> dict[str, np.ndarray]:
    """Returns image uint8 [H,W,3] RGB, mask, depression (m), valid (float32 [H,W])."""
    rng = np.random.default_rng(seed)
    f = size * rng.uniform(.75, 1.05); pitch = np.deg2rad(rng.uniform(30, 52)); h = rng.uniform(1., 2.2)
    X, Z, valid = _ground(size, f, pitch, h)
    X, Z = np.where(valid, X, 0).astype(np.float32), np.where(valid, Z, 0).astype(np.float32)
    ys, xs = np.nonzero(valid & (Z < 10))
    holes = []
    for _ in range(int(rng.integers(1, 4))):
        i = int(rng.integers(len(ys))); holes.append(_Pothole(rng, X[ys[i], xs[i]], Z[ys[i], xs[i]]))
    field = lambda a, b: np.maximum.reduce([p.depth_at(a, b) for p in holes]).astype(np.float32)
    D = field(X, Z) * valid; e = .01
    Dx, Dz = (field(X + e, Z) - D) / e, (field(X, Z + e) - D) / e
    n = np.stack([Dx, np.ones_like(D), Dz]); n /= np.linalg.norm(n, axis=0)
    el, az = np.deg2rad(rng.uniform(25, 80)), rng.uniform(0, 6.28)
    l = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    sun = rng.uniform(.1, .6)
    light = (1 - sun) + sun * np.clip(np.tensordot(l, n, 1), 0, 1) / l[1]
    ao = np.exp(-D * rng.uniform(5, 15))
    # distractor shadows (not potholes)
    for _ in range(int(rng.integers(0, 4))):
        xc, zc = rng.uniform(-3, 3), rng.uniform(2, 15); a, b = rng.uniform(.4, 2.5, 2); th = rng.uniform(0, 3.14)
        u = (np.cos(th) * (X - xc) + np.sin(th) * (Z - zc)) / a; w = (-np.sin(th) * (X - xc) + np.cos(th) * (Z - zc)) / b
        light = light * (1 - rng.uniform(.3, .6) * np.exp(-(u * u + w * w) ** rng.uniform(1.5, 4)))
    # albedo
    albedo = rng.uniform(.22, .45) * (1 + .25 * _noise(rng, X, Z, 3) + .15 * _noise(rng, X, Z, 15) + .2 * _noise(rng, X, Z, 60))
    if rng.random() < .4:  # patch
        xc, zc = rng.uniform(-2, 2), rng.uniform(2, 12); m = (abs(X - xc) < rng.uniform(.3, 1.2)) & (abs(Z - zc) < rng.uniform(.3, 1.5))
        albedo = np.where(m, albedo * rng.uniform(.6, .85), albedo)
    if rng.random() < .4:  # lane marking
        xl = rng.uniform(-2, 2); m = (abs(X - xl) < .08) & ((Z % 6) < 3)
        albedo = np.where(m, .8, albedo)
    crack = np.zeros((size, size), np.float32)
    for _ in range(int(rng.integers(0, 6))):
        p = rng.uniform(0, size, 2); pts = [p.copy()]
        for _ in range(int(rng.integers(5, 20))):
            p = p + rng.normal(0, 14, 2); pts.append(p.copy())
        cv2.polylines(crack, [np.array(pts, np.int32)], False, 1., 1)
    albedo = albedo * (1 - .6 * cv2.GaussianBlur(crack, (0, 0), .7))
    tint = rng.uniform(.94, 1.06, 3)
    rgb = albedo[..., None] * tint * (light * ao)[..., None]
    sky = (np.linspace(0, 1, size)[:, None, None] * .15 + rng.uniform(.6, .9, 3)).astype(np.float32)
    rgb = np.where(valid[..., None], rgb, sky)
    rgb = np.clip(rgb * rng.uniform(.7, 1.4), 0, 1) ** (1 / rng.uniform(.8, 1.3))
    s = rng.uniform(0, 1.2)
    if s > .1: rgb = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), s)
    rgb = np.clip(rgb + rng.normal(0, rng.uniform(0, .03), rgb.shape), 0, 1)
    img = (rgb * 255).astype(np.uint8)
    if rng.random() < .5:
        _, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        img = cv2.imdecode(buf, 1)[..., ::-1].copy()
    return {"image": img, "mask": ((D > .01) & valid).astype(np.float32), "depression": D.astype(np.float32), "valid": valid.astype(np.float32)}
