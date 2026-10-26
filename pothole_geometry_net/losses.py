"""Modular losses: BCE+Dice, masked Huber depression, heteroscedastic NLL, road smoothness."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class PGNLoss(nn.Module):
    def __init__(self, w_seg=1., w_dep=1., w_unc=.1, w_reg=.05, pos_weight=3., dep_pothole_weight=4., huber_delta=.5):
        super().__init__()
        self.w = (w_seg, w_dep, w_unc, w_reg); self.pw, self.dpw, self.delta = pos_weight, dep_pothole_weight, huber_delta

    def forward(self, out: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        lg, m, v = out["pothole_logits"].float(), b["mask"], b["valid"]
        pred, tgt = out["depression"].float(), b["depression"]
        bce = F.binary_cross_entropy_with_logits(lg, m, pos_weight=torch.tensor(self.pw, device=lg.device))
        p = torch.sigmoid(lg)
        dice = 1 - (2 * (p * m).flatten(1).sum(1) + 1) / (p.flatten(1).sum(1) + m.flatten(1).sum(1) + 1)
        l_seg = bce + dice.mean()
        wt = v * (1 + self.dpw * m); den = wt.sum().clamp(min=1.)
        l_dep = (wt * F.huber_loss(pred, tgt, delta=self.delta, reduction="none")).sum() / den
        s = out["log_variance"].float().clamp(-6, 4)  # Gaussian NLL; prediction detached so it only trains the variance head
        l_unc = (wt * .5 * (torch.exp(-s) * (pred.detach() - tgt) ** 2 + s)).sum() / den
        clean = v * (1 - m)  # smoothness only on undamaged road
        cx, cy = clean[..., :, 1:] * clean[..., :, :-1], clean[..., 1:, :] * clean[..., :-1, :]
        l_reg = ((pred[..., :, 1:] - pred[..., :, :-1]).abs() * cx).sum() / cx.sum().clamp(min=1.) + \
                ((pred[..., 1:, :] - pred[..., :-1, :]).abs() * cy).sum() / cy.sum().clamp(min=1.)
        total = self.w[0] * l_seg + self.w[1] * l_dep + self.w[2] * l_unc + self.w[3] * l_reg
        return {"total": total, "seg": l_seg.detach(), "dep": l_dep.detach(), "unc": l_unc.detach(), "reg": l_reg.detach()}
