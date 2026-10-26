"""PotholeGeometryNet: compact depthwise-separable CNN encoder-decoder."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, c)


class DSBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 + GroupNorm + SiLU (residual if shapes match)."""
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.norm, self.act = _gn(cout), nn.SiLU(inplace=True)
        self.res = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm(self.pw(self.dw(x))))
        return x + y if self.res else y


class PotholeGeometryNet(nn.Module):
    """Input [B,3,H,W] (H,W multiples of 16) -> dict of [B,1,H,W] maps:
    pothole_logits, depression (>=0, normalized relative units), log_variance."""
    def __init__(self, channels: tuple[int, ...] = (24, 48, 96, 160, 256)):
        super().__init__()
        c1, c2, c3, c4, cb = channels
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, 2, 1, bias=False), _gn(c1), nn.SiLU(inplace=True), DSBlock(c1, c1))
        self.s2 = nn.Sequential(DSBlock(c1, c2, 2), DSBlock(c2, c2))
        self.s3 = nn.Sequential(DSBlock(c2, c3, 2), DSBlock(c3, c3))
        self.s4 = nn.Sequential(DSBlock(c3, c4, 2), DSBlock(c4, c4))
        self.bott = nn.Sequential(DSBlock(c4, cb), DSBlock(cb, cb))
        self.d4 = DSBlock(cb + c4, 96)
        self.d3 = DSBlock(96 + c3, 64)
        self.d2 = DSBlock(64 + c2, 40)
        self.d1 = DSBlock(40 + c1, 32)
        self.full = nn.Sequential(nn.Conv2d(3, 8, 3, 1, 1, bias=False), _gn(8), nn.SiLU(inplace=True))
        self.refine = DSBlock(32 + 8, 24)
        self.seg, self.dep, self.var = (nn.Conv2d(24, 1, 1) for _ in range(3))

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        f1 = self.stem(img); f2 = self.s2(f1); f3 = self.s3(f2); f4 = self.s4(f3)
        x = self.bott(f4)
        x = self.d4(torch.cat([x, f4], 1))
        x = self.d3(torch.cat([self._up(x, f3), f3], 1))
        x = self.d2(torch.cat([self._up(x, f2), f2], 1))
        x = self.d1(torch.cat([self._up(x, f1), f1], 1))
        x = self.refine(torch.cat([self._up(x, img), self.full(img)], 1))
        return {"pothole_logits": self.seg(x), "depression": F.softplus(self.dep(x)), "log_variance": self.var(x)}


if __name__ == "__main__":
    m = PotholeGeometryNet()
    print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    print({k: tuple(v.shape) for k, v in m(torch.zeros(1, 3, 512, 512)).items()})
