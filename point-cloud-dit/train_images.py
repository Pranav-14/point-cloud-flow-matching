"""Overfit DiT or PTv3 flow matching on 10 images.

Procedure: all images in every step, one label per image, AdamW lr 3e-4, gradient
clip 1.0, EMA 0.99, and sampling from the EMA model with Heun, 50 steps.

Two errors are reported, on pixels in [-1, 1]:
    paired MSE   each sample against the image of its own label
    nearest MSE  each sample against its closest training image

Example:
    python train_images.py --model ptv3 --pred x --loss-space v
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from flow import PRED_TYPES, FlowMatching
from image_models import ImageDiT, ImagePTv3


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dit", "ptv3"], default="ptv3")
    p.add_argument("--pred", choices=PRED_TYPES, default="x")
    p.add_argument("--loss-space", choices=PRED_TYPES, default=None, help="default: same as --pred")
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--data-root", default="~/fastdata/cifar10")
    p.add_argument("--attn-patch", type=int, default=64, help="PTv3: tokens per attention patch")
    p.add_argument("--knn", type=int, default=8, help="PTv3: neighbours in the local encoding")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--final-steps", type=int, nargs="+", default=[50, 200], help="solver steps of the final evaluation")
    p.add_argument("--final-seeds", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    args.loss_space = args.loss_space or args.pred
    args.out = args.out or f"runs/images/{args.model}_{args.pred}-pred_{args.loss_space}-loss"
    return args


def load_images(n, image_size, root):
    """The first n CIFAR-10 training images, (n, 3, H, W) in [-1, 1]."""
    tf = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    ds = datasets.CIFAR10(root=os.path.expanduser(root), train=True, download=True, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)]) * 2 - 1


def build_model(args):
    if args.model == "ptv3":
        return ImagePTv3(args.image_size, args.patch_size, args.n_images, args.attn_patch, args.knn)
    return ImageDiT(args.image_size, args.patch_size, args.n_images)


def generate(fm, ema, images, labels, steps, method, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    net = lambda z, t: ema(z, t, labels)
    return fm.sample(net, images.shape, steps=steps, method=method, device=device, generator=g).clamp(-1, 1)


def errors(samples, images):
    paired = (samples - images).pow(2).flatten(1).mean(1)
    a, b = samples.flatten(1), images.flatten(1)
    nearest = (torch.cdist(a, b) ** 2 / a.shape[1]).min(dim=1).values
    return paired, nearest


def run(args, images):
    torch.manual_seed(args.seed)
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    images = images.to(device)
    labels = torch.arange(len(images), device=device)
    model = build_model(args).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    fm = FlowMatching(pred=args.pred, loss_space=args.loss_space)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} | pred={args.pred} loss={args.loss_space} | {n_params / 1e6:.2f}M params | device={device}")

    log = {"step": [], "loss": [], "eval_step": [], "paired_mse": [], "nearest_mse": []}
    start = time.time()
    for step in range(1, args.steps + 1):
        loss = fm.loss(lambda z, t: model(z, t, labels), images)
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
            print(f"step {step:6d} | loss {loss.item():.4f} | {time.time() - start:.0f}s")
        if step % args.eval_every == 0 or step == args.steps:
            samples = generate(fm, ema, images, labels, 50, "heun", 0, device)
            paired, nearest = errors(samples, images)
            log["eval_step"].append(step)
            log["paired_mse"].append(paired.mean().item())
            log["nearest_mse"].append(nearest.mean().item())
            save_image(torch.cat([images, samples]) * 0.5 + 0.5, out / f"samples_{step:06d}.png", nrow=len(images), padding=1)
            print(f"  [eval @ {step}] paired MSE {paired.mean():.4f} | nearest MSE {nearest.mean():.4f}")

    print(f"{'solver':>10s} {'paired':>9s} {'worst img':>10s} {'nearest':>9s}")
    final = {}
    for method in ("heun", "euler"):
        for steps in args.final_steps:
            results = [errors(generate(fm, ema, images, labels, steps, method, s, device), images) for s in range(args.final_seeds)]
            paired = torch.stack([r[0] for r in results]).mean(0)
            nearest = torch.stack([r[1] for r in results]).mean(0)
            name = f"{method}/{steps}"
            final[name] = {"paired": paired.mean().item(), "worst": paired.max().item(), "nearest": nearest.mean().item()}
            print(f"{name:>10s} {paired.mean():9.4f} {paired.max():10.4f} {nearest.mean():9.4f}")

    log["final"] = final
    (out / "log.json").write_text(json.dumps(log, indent=2))
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "args": vars(args)}, out / "model.pt")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(log["step"], log["loss"])
    ax[0].set(title=f"training loss ({args.loss_space}-space)", xlabel="step", yscale="log")
    ax[1].plot(log["eval_step"], log["paired_mse"], marker="o", label="paired")
    ax[1].plot(log["eval_step"], log["nearest_mse"], marker="o", label="nearest")
    ax[1].set(title="sample MSE", xlabel="step", yscale="log")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)
    print(f"done -> {out}")
    return log


if __name__ == "__main__":
    args = get_args()
    run(args, load_images(args.n_images, args.image_size, args.data_root))
