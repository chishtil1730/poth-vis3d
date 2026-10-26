import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model import PotholeGeometryNet
from losses import PGNLoss
from dataset import SyntheticPotholes


def test_shapes_finite():
    o = PotholeGeometryNet()(torch.randn(2, 3, 64, 64))
    for v in o.values(): assert v.shape == (2, 1, 64, 64) and torch.isfinite(v).all()
    assert (o["depression"] >= 0).all()


def test_loss_backward():
    b = SyntheticPotholes(2, 0, 64)[0]; b = {k: v[None] for k, v in b.items()}
    m = PotholeGeometryNet(); l = PGNLoss()(m(b["image"]), b)["total"]; l.backward(); assert torch.isfinite(l)
