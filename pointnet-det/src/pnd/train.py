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

from .boxes import (NUM_HEADING_BINS, BIN_WIDTH, corner_loss,
                    decode_heading, anchor_diagonal)
from .config import Config
from .dataset import loaders
from .kitti import CLASSES
from .model import build


def box_loss(out, batch, fg, corner_w: float = 1.0):
    """Direct centre/size/heading supervision, with corner loss as a REGULARISER.

    An earlier version replaced the direct centre and size terms with corner loss
    alone. That was a misreading of F-PointNet, where the loss is

        L_centre-reg + L_size-cls/reg + L_heading-cls/reg + gamma * L_corner

    -- corner loss is added ON TOP of the direct terms, not instead of them.
    Dropping them left size supervised only weakly and entangled through the
    corners, and dividing corner loss by the anchor diagonal further shrank the
    gradient for the largest anchor. Measured cost: Car moderate AP fell
    26.20 -> 19.11 while heading error improved 17.4 -> 6.4 degrees. Heading was
    fixed; size and centre regressed further than heading gained.

    Returns (centre, size, heading-bin, heading-residual, corner).
    """
    z = out["center"].sum() * 0.0
    if fg.sum() == 0:
        return z, z, z, z, z

    l_ctr = F.smooth_l1_loss(out["center"][fg], batch["center"][fg])
    l_size = F.smooth_l1_loss(out["size_log"][fg], batch["size_log"][fg])

    hb_gt = batch["head_bin"][fg]
    l_bin = F.cross_entropy(out["head_bin"][fg], hb_gt)
    res_at_gt = out["head_res"][fg].gather(1, hb_gt.unsqueeze(1)).squeeze(1)
    l_res = F.smooth_l1_loss(res_at_gt, batch["head_res"][fg])

    anchor = batch["anchor"][fg]
    scale = batch["scale"][fg].unsqueeze(1)
    dims_pred = torch.exp(out["size_log"][fg].clamp(-2.0, 2.0)) * anchor
    yaw_pred = decode_heading(hb_gt.float(), res_at_gt)
    yaw_gt = decode_heading(hb_gt.float(), batch["head_res"][fg])
    l_corner = corner_w * corner_loss(
        out["center"][fg] * scale, dims_pred, yaw_pred,
        batch["center"][fg] * scale, batch["dims"][fg], yaw_gt,
        anchor_diagonal(anchor))
    return l_ctr, l_size, l_bin, l_res, l_corner


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, num_classes):
    model.eval()
    conf = np.zeros((num_classes, num_classes), np.int64)
    ctr_err, yaw_err, n_fg = 0.0, 0.0, 0
    per = {c: [0.0, 0.0, 0.0, 0] for c in range(1, num_classes)}  # ctr, size, yaw, n
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
            hb = out["head_bin"][fg].float().argmax(1)
            hr = out["head_res"][fg].float().gather(1, hb.unsqueeze(1)).squeeze(1)
            yp = decode_heading(hb.float(), hr)
            yg = decode_heading(b["head_bin"].to(device)[fg].float(),
                                b["head_res"].to(device)[fg])
            # fold to [0, 90]: a box rotated by pi is the same box
            d = torch.remainder(yp - yg, np.pi)
            d = torch.minimum(d, np.pi - d)
            yaw_err += torch.rad2deg(d).sum().item()
            n_fg += int(fg.sum())

            # per-class breakdown: which term is actually costing IoU?
            ce = ((out["center"][fg].float() - b["center"].to(device)[fg])
                  .norm(dim=1) * sc)
            dp = torch.exp(out["size_log"][fg].float().clamp(-2, 2))                 * b["anchor"].to(device)[fg]
            se = (dp - b["dims"].to(device)[fg]).abs().mean(dim=1)
            ye = torch.rad2deg(d)
            yc = y[fg]
            for c in per:
                m = yc == c
                if m.any():
                    per[c][0] += ce[m].sum().item()
                    per[c][1] += se[m].sum().item()
                    per[c][2] += ye[m].sum().item()
                    per[c][3] += int(m.sum())
    return conf, (ctr_err / max(n_fg, 1)), (yaw_err / max(n_fg, 1)), per


def report(conf, ctr_err, yaw_err, per=None) -> dict:
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

    # Per-class box error. Aggregate centre/heading numbers hid the fact that a
    # 0.33 m centre error is 8% of a car's length but half a pedestrian's width,
    # and that size was left effectively unsupervised when corner loss replaced
    # the direct terms. Break it out so the binding term is visible.
    box_err = {}
    if per:
        lines.append(f"  {'':<12}{'ctr m':>9}{'size m':>9}{'yaw deg':>9}")
        for i, c in enumerate(CLASSES):
            if i == 0 or i not in per or per[i][3] == 0:
                continue
            n = per[i][3]
            box_err[c] = {"ctr": per[i][0] / n, "size": per[i][1] / n,
                          "yaw": per[i][2] / n}
            lines.append(f"  {c:<12}{per[i][0]/n:>9.2f}{per[i][1]/n:>9.2f}"
                         f"{per[i][2]/n:>9.1f}")

    return {"text": "\n".join(lines), "fg_recall": float(fg_rec),
            "f1_fg": float(f1[1:].mean()), "ctr_err": float(ctr_err),
            "yaw_err": float(yaw_err), "box_err": box_err,
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

    # Inverse frequency gives [0.28, 2.57, 19.2, 20] and buys recall with false
    # positives: measured Car precision 0.86 against recall 0.98, Pedestrian
    # 0.65 against 0.92. sqrt-inverse is the gentler standard alternative.
    #
    # Normalised so the EXPECTED weight over the data is 1 (sum(w * freq) = 1).
    # Dividing by w.mean() instead shrinks the classification loss by ~10x
    # relative to the box loss, which then dominates.
    freq = np.maximum(counts, 1) / counts.sum()
    if cfg.weight_mode == "inv":
        w = 1.0 / freq
    elif cfg.weight_mode == "sqrt_inv":
        w = 1.0 / np.sqrt(freq)
    elif cfg.weight_mode == "none":
        w = np.ones_like(freq)
    else:
        raise ValueError(f"weight_mode must be inv|sqrt_inv|none, got {cfg.weight_mode!r}")
    w = w / float((w * freq).sum())
    w = np.clip(w, 1.0 / cfg.weight_clip, cfg.weight_clip)
    print(f"class weights ({cfg.weight_mode}):", np.round(w, 3).tolist(), "\n")
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

    # Exponential moving average of the weights. Cheap, and it removes most of
    # the epoch-to-epoch thrash seen in the first ablation, where Pedestrian F1
    # swung between 0.28 and 0.68 on a 514-sample validation slice.
    ema = None
    if cfg.ema_decay > 0:
        import copy
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def ema_update():
        if ema is None:
            return
        d = cfg.ema_decay
        for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
            if pe.dtype.is_floating_point:
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            else:
                pe.copy_(pm)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = cfg.run_dir / (cfg.tag or cfg.canon)
    out_dir.mkdir(parents=True, exist_ok=True)

    best, history = -1.0, []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in tl:
            x = b["x"].to(cfg.device, non_blocking=True)
            y = b["cls"].to(cfg.device, non_blocking=True)
            for k in ("center", "size_log", "head_bin", "head_res",
                      "dims", "anchor", "scale"):
                b[k] = b[k].to(cfg.device, non_blocking=True)

            with torch.autocast(cfg.device,
                                dtype=getattr(torch, cfg.amp_dtype)
                                if cfg.amp_dtype else torch.float32,
                                enabled=cfg.amp_dtype is not None):
                out = model(x)
                lcls = F.cross_entropy(out["logits"], y, weight=wt,
                                       label_smoothing=cfg.label_smoothing)
                fg = y > 0
                lc, lsz, lbin, lres, lcorn = box_loss(out, b, fg,
                                                      cfg.corner_loss_w)
                loss = lcls + cfg.box_loss_w * (lc + lsz + 0.5 * lbin + lres
                                                + lcorn)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema_update()
            tot += float(loss.detach())
            nb += 1

        conf, ce, ye, per = evaluate(model, vl, cfg.device, cfg.amp_dtype,
                                     cfg.num_classes)
        rep = report(conf, ce, ye, per)
        rep["src"] = "raw"
        if ema is not None:
            c2, ce2, ye2, per2 = evaluate(ema, vl, cfg.device, cfg.amp_dtype,
                                          cfg.num_classes)
            r2 = report(c2, ce2, ye2, per2)
            if r2["f1_fg"] > rep["f1_fg"]:
                r2["src"] = "ema"
                rep = r2
        el = time.time() - t0
        print(f"epoch {ep+1:>3}/{cfg.epochs}  loss {tot/max(nb,1):.4f}  "
              f"{el:.0f}s  [{rep['src']}]")
        print(rep["text"])
        history.append({"epoch": ep + 1, "loss": tot / max(nb, 1), **{
            k: v for k, v in rep.items() if k not in ("text", "src")}})

        if rep["f1_fg"] > best:
            best = rep["f1_fg"]
            keep = ema if (ema is not None and rep["src"] == "ema") else model
            torch.save({"model": keep.state_dict(), "cfg": cfg.to_dict(),
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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None,
                    help="run directory name; defaults to the canon name. "
                         "Set it for multi-seed runs, or every seed writes "
                         "best.pt and history.json to the same place.")
    a = ap.parse_args()
    cfg = Config.load(a.config, canon=a.canon, epochs=a.epochs, lr=a.lr,
                      num_workers=a.num_workers, seed=a.seed, tag=a.tag)
    if a.batch_size:
        cfg.batch_size = a.batch_size
    if a.no_compile:
        cfg.compile = False
    train(cfg)


if __name__ == "__main__":
    main()
