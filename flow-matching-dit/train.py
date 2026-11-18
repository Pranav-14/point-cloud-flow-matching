"""Overfit a DiT flow-matching model on a few images.

Example:
    python train.py --pred v
    python train.py --pred x  --loss-space v
    python train.py --pred eps --loss-space eps
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from flow import PRED_TYPES, FlowMatching
from utils import build_model, load_images, nearest_train_mse, save_grid


def get_args():
    p = argparse.ArgumentParser()
    # flow matching
    p.add_argument("--pred", choices=PRED_TYPES, default="v", help="what the network outputs")
    p.add_argument("--loss-space", choices=PRED_TYPES, default=None, help="space of the MSE loss (default: same as --pred)")
    p.add_argument("--t-sampler", choices=["uniform", "logit_normal"], default="uniform")
    # data
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--data-root", default="~/fastdata/cifar10", help="where CIFAR-10 is stored/downloaded")
    # model
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    # optimisation
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    # logging / sampling
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--out", default=None, help="output dir (default: runs/<pred>-pred_<loss>-loss)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.loss_space = args.loss_space or args.pred
    args.out = args.out or f"runs/{args.pred}-pred_{args.loss_space}-loss"
    return args


def sample_and_report(fm, model, args, train, out_dir, tag):
    model.eval()
    g = torch.Generator(device=args.device).manual_seed(1234)  # same noise every time -> comparable grids
    shape = (args.n_samples, 3, args.image_size, args.image_size)
    samples = fm.sample(model, shape, steps=args.sample_steps, device=args.device, generator=g)
    model.train()
    save_grid(samples, out_dir / f"samples_{tag}.png")
    mse, _ = nearest_train_mse(samples, train)
    return mse.mean().item()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    train = load_images(args.n_images, args.image_size, args.data_root).to(args.device)
    save_grid(train, out_dir / "train_images.png")

    model = build_model(args).to(args.device)
    fm = FlowMatching(pred=args.pred, loss_space=args.loss_space, t_sampler=args.t_sampler)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"pred={args.pred} loss={args.loss_space} | {n_params / 1e6:.2f}M params | device={args.device}")

    log = {"step": [], "loss": [], "sample_step": [], "nn_mse": []}
    running, t0 = 0.0, time.time()
    for step in range(1, args.steps + 1):
        x = train[torch.randint(len(train), (args.batch_size,), device=args.device)]
        loss = fm.loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += loss.item()
        if step % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            log["step"].append(step)
            log["loss"].append(avg)
            print(f"step {step:6d} | loss {avg:.4f} | {time.time() - t0:.0f}s")

        if step % args.sample_every == 0 or step == args.steps:
            nn_mse = sample_and_report(fm, model, args, train, out_dir, f"{step:06d}")
            log["sample_step"].append(step)
            log["nn_mse"].append(nn_mse)
            print(f"  samples saved | mean MSE to nearest train image: {nn_mse:.4f}")

    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "model.pt")
    (out_dir / "log.json").write_text(json.dumps(log))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(log["step"], log["loss"])
    ax[0].set(title=f"training loss ({args.loss_space}-space)", xlabel="step", yscale="log")
    ax[1].plot(log["sample_step"], log["nn_mse"], marker="o")
    ax[1].set(title="samples: MSE to nearest train image", xlabel="step", yscale="log")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=120)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
