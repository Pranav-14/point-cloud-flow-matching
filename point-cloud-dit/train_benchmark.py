"""Train DiT or PTv3 on the shape benchmark.

Overfit check (one fixed clean cloud per shape, 3000 steps):
    python train_benchmark.py --model ptv3 --pred x --overfit 8

Full comparison (fresh augmented clouds, 10000 steps):
    python train_benchmark.py --model ptv3 --pred x
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from benchmark_data import DEFAULT_SHAPES, ShapeBank, augmentation
from dit import PointDiT
from flow import PRED_TYPES, FlowMatching
from metrics import nearest_same_class
from ptv3 import PointPTv3
from utils import chamfer, save_clouds


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dit", "ptv3"], default="dit")
    p.add_argument("--pred", choices=PRED_TYPES, default="x")
    p.add_argument("--loss-space", choices=PRED_TYPES, default="v")
    # data
    p.add_argument("--shapes", nargs="+", default=list(DEFAULT_SHAPES))
    p.add_argument("--n-points", type=int, default=256)
    p.add_argument("--no-rot", action="store_true")
    p.add_argument("--scale-min", type=float, default=0.6)
    p.add_argument("--scale-max", type=float, default=1.0)
    p.add_argument("--trans", type=float, default=0.3)
    p.add_argument("--jitter", type=float, default=0.01)
    # model
    p.add_argument("--fourier", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256, help="DiT only")
    p.add_argument("--depth", type=int, default=8, help="DiT only")
    p.add_argument("--heads", type=int, default=4, help="DiT only")
    p.add_argument("--patch", type=int, default=64, help="PTv3 only")
    p.add_argument("--knn", type=int, default=8, help="PTv3 only")
    # training
    p.add_argument("--overfit", type=int, default=0, help="if > 0: first N shapes, one fixed clean cloud each")
    p.add_argument("--steps", type=int, default=None, help="default: 3000 overfit, 10000 full")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default: 3e-4 overfit, 1e-4 full")
    p.add_argument("--ema", type=float, default=None, help="default: 0.99 overfit, 0.999 full")
    p.add_argument("--label-drop", type=float, default=0.1, help="full training only")
    p.add_argument("--val-per-class", type=int, default=64)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--no-clamp", action="store_true", help="do not clip samples to [-1, 1]")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    overfit = args.overfit > 0
    args.steps = args.steps or (3000 if overfit else 10000)
    args.lr = args.lr or (3e-4 if overfit else 1e-4)
    args.ema = args.ema or (0.99 if overfit else 0.999)
    if args.out is None:
        mode = "overfit" if overfit else "full"
        args.out = f"runs/benchmark/{mode}/{args.model}_{args.pred}-pred_{args.loss_space}-loss_seed{args.seed}"
    return args


def build_model(args, n_classes):
    if args.model == "ptv3":
        return PointPTv3(
            fourier=args.fourier, patch=args.patch, k=args.knn, n_classes=n_classes,
            enc_dims=(64, 128, 256, 512), enc_heads=(2, 4, 8, 16), dec_dims=(64, 128, 256), dec_heads=(2, 4, 8),
        )
    return PointDiT(hidden=args.hidden, depth=args.depth, heads=args.heads, fourier=args.fourier, n_classes=n_classes)


def generate(fm, model, labels, args, steps=None, method="heun", seed=0):
    g = torch.Generator(device=args.device).manual_seed(seed)
    shape = (len(labels), args.n_points, 2)
    net = lambda z, t: model(z, t, labels)
    x = fm.sample(net, shape, steps=steps or args.sample_steps, method=method, device=args.device, generator=g)
    return x if args.no_clamp else x.clamp(-1, 1)


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    device, overfit = args.device, args.overfit > 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    bank = ShapeBank(args.shapes[: args.overfit] if overfit else args.shapes, args.n_points, device)
    C, aug = bank.n_classes, augmentation(args)
    data_g = torch.Generator(device=device).manual_seed(args.seed)
    if overfit:
        train_x, train_y = bank.fixed(1, seed=0)
        floor = chamfer(train_x, bank.fixed(1, seed=1)[0]).mean().item()
        print(f"[overfit] {C} clean clouds | Chamfer floor {floor:.5f}")
    else:
        val_x, val_y = bank.fixed(args.val_per_class, seed=1234, augment=True, **aug)
        real_x, real_y = bank.fixed(6, seed=999, augment=True, **aug)
        floor = nearest_same_class(real_x, real_y, val_x, val_y)[0].mean().item()
        print(f"[data] {C} shapes | {len(val_x)} validation clouds | real-cloud floor {floor:.5f}")

    model = build_model(args, C).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    fm = FlowMatching(pred=args.pred, loss_space=args.loss_space)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} | pred={args.pred} loss={args.loss_space} | {n_params / 1e6:.2f}M params | "
          f"lr={args.lr} ema={args.ema} | device={device}")

    log = {"step": [], "loss": [], "eval_step": [], "chamfer": [], "floor": floor}
    start = time.time()
    for step in range(1, args.steps + 1):
        x, y = (train_x, train_y) if overfit else bank.batch(args.batch_size, data_g, **aug)
        labels = y
        if not overfit and args.label_drop > 0:
            drop = torch.rand(len(y), device=device) < args.label_drop
            labels = torch.where(drop, torch.full_like(y, C), y)
        loss = fm.loss(lambda z, t: model(z, t, labels), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), model.parameters()):
                e.lerp_(p, 1 - args.ema)

        if step % args.log_every == 0:
            log["step"].append(step)
            log["loss"].append(loss.item())
            print(f"step {step:6d} | loss {loss.item():.4f} | {time.time() - start:.0f}s", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            if overfit:
                samples = generate(fm, ema, train_y, args)
                d = chamfer(samples, train_x)
                save_clouds(torch.cat([train_x, samples]), out / f"samples_{step:06d}.png", ncols=C)
                print(f"  [sample @ {step}] Chamfer to own cloud: mean {d.mean():.5f} worst {d.max():.5f} (floor {floor:.5f})")
            else:
                labels_s = torch.arange(C, device=device).repeat_interleave(6)
                samples = generate(fm, ema, labels_s, args)
                d = nearest_same_class(samples, labels_s, val_x, val_y)[0]
                save_clouds(samples, out / f"samples_{step:06d}.png", ncols=6)
                print(f"  [sample @ {step}] nearest same-class Chamfer {d.mean():.5f} (floor {floor:.5f})")
            log["eval_step"].append(step)
            log["chamfer"].append(d.mean().item())

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "args": vars(args), "shapes": bank.names}, out / "ckpt.pt")
    (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
