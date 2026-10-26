"""Train PotholeGeometryNet on synthetic data. Usage: python train.py [--config config.yaml] [--smoke] [--resume ckpt]"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import cv2, numpy as np, torch, yaml
from torch.utils.data import DataLoader
from dataset import SyntheticPotholes, denorm
from losses import PGNLoss
from model import PotholeGeometryNet
from utils import colorize, hstack


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def stats(out, b) -> dict[str, float]:
    p = (torch.sigmoid(out["pothole_logits"].float()) > .5).float(); m, v = b["mask"], b["valid"]
    err = (out["depression"].float() - b["depression"]).abs() * v
    return dict(inter=(p * m).sum().item(), union=((p + m) > 0).float().sum().item(), ps=p.sum().item(), ms=m.sum().item(),
                ae=err.sum().item(), nv=v.sum().item(), aep=(err * m).sum().item(), npx=(m * v).sum().item())


def summarize(s: dict[str, float]) -> dict[str, float]:
    return {"iou": s["inter"] / max(s["union"], 1), "dice": 2 * s["inter"] / max(s["ps"] + s["ms"], 1),
            "mae_norm": s["ae"] / max(s["nv"], 1), "mae_pothole_norm": s["aep"] / max(s["npx"], 1)}


def save_panel(path: Path, b, out, k: int = 3) -> None:
    rows = []
    for i in range(min(k, b["image"].shape[0])):
        rgb = denorm(b["image"][i].cpu())[..., ::-1]; vmax = max(float(b["depression"][i].max()), 1e-3)
        t, pd = b["depression"][i, 0].cpu().numpy(), out["depression"][i, 0].float().cpu().numpy()
        pr = torch.sigmoid(out["pothole_logits"][i, 0].float()).cpu().numpy(); u = np.exp(.5 * out["log_variance"][i, 0].float().cpu().numpy())
        rows.append(hstack([rgb, (b["mask"][i, 0].cpu().numpy() * 255).astype(np.uint8), (pr * 255).astype(np.uint8),
                            colorize(t, 0, vmax), colorize(pd, 0, vmax), colorize(np.abs(pd - t), 0, vmax), colorize(u, 0, float(u.max()) + 1e-6)]))
    cv2.imwrite(str(path), np.vstack(rows))  # cols: rgb|tgt mask|pred prob|tgt dep|pred dep|abs err|uncertainty


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    if a.smoke:
        cfg.update(image_size=128, train_samples=16, val_samples=8, epochs=2, batch_size=4, workers=0, out_dir="outputs/smoke")
    set_seed(cfg["seed"]); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = cfg["amp"] and dev.type == "cuda"
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    tr = SyntheticPotholes(cfg["train_samples"], 0, cfg["image_size"], cfg["dep_scale"])
    va = SyntheticPotholes(cfg["val_samples"], 10_000_000, cfg["image_size"], cfg["dep_scale"])
    mk = lambda ds, sh: DataLoader(ds, cfg["batch_size"], shuffle=sh, num_workers=cfg["workers"], pin_memory=dev.type == "cuda", drop_last=sh)
    model = PotholeGeometryNet(tuple(cfg["model"]["channels"])).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | device {dev} | amp {amp}")
    crit = PGNLoss(**cfg["loss"]); opt = torch.optim.AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"]); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, bad = 0, float("inf"), 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev); model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); start, best = ck["epoch"] + 1, ck["best"]
    for ep in range(start, cfg["epochs"]):
        model.train(); tr.offset = ep * cfg["train_samples"]; tl = 0.; nb = 0
        for b in mk(tr, True):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(dev.type, enabled=amp):
                out = model(b["image"])
            loss = crit(out, b)["total"]
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1
        sched.step(); model.eval(); agg: dict[str, float] = {}; vl = 0.; vn = 0; first = None
        with torch.inference_mode():
            for b in mk(va, False):
                b = {k: v.to(dev) for k, v in b.items()}; out = model(b["image"]); l = crit(out, b)
                vl += l["total"].item(); vn += 1
                for k, x in stats(out, b).items(): agg[k] = agg.get(k, 0.) + x
                if first is None: first = (b, out)
        vl /= max(vn, 1); rec = {"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, **summarize(agg)}
        print(json.dumps(rec)); open(out_dir / "log.jsonl", "a").write(json.dumps(rec) + "\n")
        ck = {"model_state_dict": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "epoch": ep, "best": min(best, vl), "config": cfg}
        torch.save(ck, out_dir / "last.pt")
        if vl < best:
            best, bad = vl, 0; torch.save(ck, out_dir / "best.pt"); save_panel(out_dir / "val_panel.png", *first)
        else:
            bad += 1
            if bad >= cfg["patience"]: print("early stop"); break


if __name__ == "__main__":
    main()
