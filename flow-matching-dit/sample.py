"""Generate images from a trained checkpoint.

    python sample.py runs/v-pred_v-loss/model.pt --n 64 --steps 50 --method heun
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from flow import FlowMatching
from utils import build_model, save_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--method", choices=["euler", "heun"], default="euler")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    train_args = SimpleNamespace(**ckpt["args"])
    model = build_model(train_args).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    fm = FlowMatching(pred=train_args.pred, loss_space=train_args.loss_space, t_sampler=train_args.t_sampler)

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    shape = (args.n, 3, train_args.image_size, train_args.image_size)
    samples = fm.sample(model, shape, steps=args.steps, method=args.method, device=args.device, generator=g)

    out = Path(args.ckpt).parent / f"gen_{args.method}{args.steps}_seed{args.seed}.png"
    save_grid(samples, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
