"""Score a checkpoint from train_benchmark.py.

    python evaluate_benchmark.py --ckpt runs/benchmark/full/ptv3_x-pred_v-loss_seed0/ckpt.pt
"""
import argparse
import json
from pathlib import Path

import torch

from benchmark_data import ShapeBank, augmentation
from flow import FlowMatching
from metrics import class_accuracy, distribution_metrics, emd, nearest_same_class
from train_benchmark import build_model, generate
from utils import chamfer


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-per-class", type=int, default=128)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seeds", type=int, default=3, help="overfit checkpoints: noise seeds")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opts = p.parse_args(argv)

    ck = torch.load(opts.ckpt, map_location="cpu")
    args = argparse.Namespace(**ck["args"])
    args.device = opts.device
    names = ck["shapes"]
    bank = ShapeBank(names, args.n_points, opts.device)
    C = bank.n_classes
    model = build_model(args, C).to(opts.device)
    model.load_state_dict(ck["ema"])
    model.eval()
    fm = FlowMatching(pred=args.pred, loss_space=args.loss_space)
    print(f"{opts.ckpt}: {args.model}, pred={args.pred}, loss={args.loss_space}")

    def sample(labels, steps, method="heun", seed=0):
        parts = [generate(fm, model, labels[i:i + opts.batch_size], args, steps, method, seed + i)
                 for i in range(0, len(labels), opts.batch_size)]
        return torch.cat(parts)

    results = {}
    if args.overfit > 0:
        x, y = bank.fixed(1, seed=0)
        floor = chamfer(x, bank.fixed(1, seed=1)[0]).mean().item()
        print(f"Chamfer floor {floor:.5f}\n{'solver':>10s} {'CD mean':>10s} {'CD worst':>10s} {'EMD mean':>10s}")
        for method in ("heun", "euler"):
            for steps in (opts.steps, 4 * opts.steps):
                runs = [sample(y, steps, method, s) for s in range(opts.seeds)]
                cd = torch.stack([chamfer(r, x).cpu() for r in runs]).mean(0)
                em = torch.stack([emd(r, x) for r in runs]).mean(0)
                name = f"{method}/{steps}"
                results[name] = {"cd_mean": cd.mean().item(), "cd_worst": cd.max().item(), "emd_mean": em.mean().item()}
                print(f"{name:>10s} {cd.mean():10.5f} {cd.max():10.5f} {em.mean():10.5f}")
    else:
        aug = augmentation(args)
        ref, ref_y = bank.fixed(opts.n_per_class, seed=1234, augment=True, **aug)
        real, real_y = bank.fixed(opts.n_per_class, seed=999, augment=True, **aug)
        y = torch.arange(C, device=opts.device).repeat_interleave(opts.n_per_class)
        gen = sample(y, opts.steps)

        def score(x, labels):
            m, per = distribution_metrics(x, labels, ref, ref_y, C)
            d, idx = nearest_same_class(x, labels, ref, ref_y)
            m.update(class_acc=class_accuracy(x, labels, ref, ref_y), nearest_cd=d.mean().item(),
                     nearest_emd=emd(x, ref[idx]).mean().item())
            return m, per

        g, per = score(gen, y)
        r, _ = score(real, real_y)
        print(f"\n{opts.n_per_class} clouds per class, Heun {opts.steps} steps")
        print(f"{'metric':24s} {'generated':>10s} {'real vs real':>13s}")
        labels = {"mmd_cd": "MMD-CD (lower better)", "cov_cd": "COV-CD (higher better)", "nna": "1-NNA (50% ideal)",
                  "class_acc": "class accuracy", "nearest_cd": "nearest-ref Chamfer", "nearest_emd": "nearest-ref EMD"}
        for key, label in labels.items():
            print(f"{label:24s} {g[key]:10.4f} {r[key]:13.4f}")
        print("per-class MMD-CD: " + ", ".join(f"{n} {v:.5f}" for n, v in zip(names, per["mmd_cd"])))
        results = {"generated": g, "real_vs_real": r, "per_class_mmd": dict(zip(names, per["mmd_cd"]))}

    (Path(opts.ckpt).parent / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
