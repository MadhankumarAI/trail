"""
Training loop.

    python -m pnd.train --canon pca2_yaw --epochs 20

Losses
------
classification  cross entropy, inverse-frequency weighted. Background outruns
                foreground about 13:1 in the cached proposals, so unweighted CE
                converges happily to "predict background" at 93% accuracy and
                zero recall.

box             smooth L1 on centre offset and log-size, plus (1 - cos) on the
                heading, applied to foreground only. Regressing a box for a
                background cluster is meaningless and the gradient is noise.

Reported per epoch: overall accuracy is deliberately NOT the headline. With this
class balance it is a useless number. Foreground recall and per-class F1 are
what move.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .dataset import loaders
from .kitti import CLASSES
from .model import build


def box_loss(out, batch, fg):
    if fg.sum() == 0:
        z = out["center"].sum() * 0.0
        return z, z, z
    lc = F.smooth_l1_loss(out["center"][fg], batch["center"][fg])
    ls = F.smooth_l1_loss(out["size_log"][fg], batch["size_log"][fg])
    ly = (1.0 - (out["yaw_sc"][fg] * batch["yaw_sc"][fg]).sum(1)).mean()
    return lc, ls, ly


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, num_classes):
    model.eval()
    conf = np.zeros((num_classes, num_classes), np.int64)
    ctr_err, yaw_err, n_fg = 0.0, 0.0, 0
    for b in loader:
        x = b["x"].to(device, non_blocking=True)
        y = b["cls"].to(device, non_blocking=True)
        with torch.autocast(device, dtype=getattr(torch, amp_dtype),
                            enabled=amp_dtype is not None):
            out = model(x)
        pred = out["logits"].argmax(1)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            conf[t, p] += 1
        fg = y > 0
        if fg.any():
            sc = b["scale"].to(device)[fg]
            ctr_err += ((out["center"][fg].float() - b["center"].to(device)[fg])
                        .norm(dim=1) * sc).sum().item()
            cs = (out["yaw_sc"][fg].float() * b["yaw_sc"].to(device)[fg]).sum(1)
            yaw_err += torch.rad2deg(torch.acos(cs.clamp(-1, 1))).sum().item()
            n_fg += int(fg.sum())
    return conf, (ctr_err / max(n_fg, 1)), (yaw_err / max(n_fg, 1))


def report(conf, ctr_err, yaw_err) -> dict:
    tp = np.diag(conf).astype(float)
    sup = conf.sum(1).astype(float)
    pred = conf.sum(0).astype(float)
    rec = tp / np.maximum(sup, 1)
    prec = tp / np.maximum(pred, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    fg_sup = sup[1:].sum()
    fg_rec = tp[1:].sum() / max(fg_sup, 1)
    lines = [f"  {'class':<12}{'support':>9}{'prec':>8}{'rec':>8}{'F1':>8}"]
    for i, c in enumerate(CLASSES):
        lines.append(f"  {c:<12}{int(sup[i]):>9,}{prec[i]:>8.3f}"
                     f"{rec[i]:>8.3f}{f1[i]:>8.3f}")
    lines.append(f"  {'-'*43}")
    lines.append(f"  foreground recall {fg_rec:.3f}   "
                 f"mean F1 (fg) {f1[1:].mean():.3f}")
    lines.append(f"  centre err {ctr_err:.2f} m   heading err {yaw_err:.1f} deg")
    return {"text": "\n".join(lines), "fg_recall": float(fg_rec),
            "f1_fg": float(f1[1:].mean()), "ctr_err": float(ctr_err),
            "yaw_err": float(yaw_err),
            "per_class_f1": {c: float(f1[i]) for i, c in enumerate(CLASSES)}}


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    tl, vl, tr_set = loaders(cfg)
    counts = tr_set.class_counts()
    print(cfg.summary())
    print(f"\ntrain {len(tr_set):,} proposals   val {len(vl.dataset):,}")
    print("class counts:", dict(zip(CLASSES, counts.tolist())))
    print(f"bg:fg = {counts[0] / max(counts[1:].sum(), 1):.1f} : 1")

    w = counts.sum() / (cfg.num_classes * np.maximum(counts, 1))
    w = np.clip(w, 0.2, 20.0)
    print("class weights:", np.round(w, 2).tolist(), "\n")
    wt = torch.tensor(w, dtype=torch.float32, device=cfg.device)

    model = build(cfg).to(cfg.device)
    print(f"model params {model.n_params()/1e6:.2f}M")
    if cfg.compile:
        # Belt and braces: config.py already gates this on the Python version,
        # but a compile failure should degrade to eager, never abort a run that
        # is otherwise fine.
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile unavailable, running eager: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * max(len(tl), 1),
        pct_start=0.3)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.amp_dtype == "float16")

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = cfg.run_dir / cfg.canon
    out_dir.mkdir(parents=True, exist_ok=True)

    best, history = -1.0, []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in tl:
            x = b["x"].to(cfg.device, non_blocking=True)
            y = b["cls"].to(cfg.device, non_blocking=True)
            for k in ("center", "size_log", "yaw_sc"):
                b[k] = b[k].to(cfg.device, non_blocking=True)

            with torch.autocast(cfg.device,
                                dtype=getattr(torch, cfg.amp_dtype)
                                if cfg.amp_dtype else torch.float32,
                                enabled=cfg.amp_dtype is not None):
                out = model(x)
                lcls = F.cross_entropy(out["logits"], y, weight=wt)
                fg = y > 0
                lc, ls, ly = box_loss(out, b, fg)
                loss = lcls + cfg.box_loss_w * (lc + ls + ly)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            nb += 1

        conf, ce, ye = evaluate(model, vl, cfg.device, cfg.amp_dtype,
                                cfg.num_classes)
        rep = report(conf, ce, ye)
        el = time.time() - t0
        print(f"epoch {ep+1:>3}/{cfg.epochs}  loss {tot/max(nb,1):.4f}  "
              f"{el:.0f}s")
        print(rep["text"])
        history.append({"epoch": ep + 1, "loss": tot / max(nb, 1), **{
            k: v for k, v in rep.items() if k != "text"}})

        if rep["f1_fg"] > best:
            best = rep["f1_fg"]
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                        "metrics": rep}, out_dir / "best.pt")
            print(f"  -> new best, saved")
        print()

    (out_dir / "history.json").write_text(json.dumps(
        {"config": cfg.to_dict(), "history": history}, indent=2))
    print(f"best foreground F1 {best:.4f}   -> {out_dir}")
    return {"canon": cfg.canon, "best_f1_fg": best, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--canon", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile even where it is supported")
    a = ap.parse_args()
    cfg = Config.load(a.config, canon=a.canon, epochs=a.epochs, lr=a.lr,
                      num_workers=a.num_workers)
    if a.batch_size:
        cfg.batch_size = a.batch_size
    if a.no_compile:
        cfg.compile = False
    train(cfg)


if __name__ == "__main__":
    main()
