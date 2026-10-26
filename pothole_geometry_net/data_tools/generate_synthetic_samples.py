"""Dump a few synthetic samples for inspection: python data_tools/generate_synthetic_samples.py outputs/synth 8"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from synthetic import render_sample
from utils import colorize, hstack
out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/synth"); out.mkdir(parents=True, exist_ok=True)
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 8):
    s = render_sample(i); np.save(out / f"{i}_depression.npy", s["depression"])
    cv2.imwrite(str(out / f"{i}.png"), hstack([s["image"][..., ::-1], (s["mask"] * 255).astype(np.uint8), colorize(s["depression"], 0, .12)]))
