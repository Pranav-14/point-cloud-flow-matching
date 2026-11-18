# Flow matching + DiT (Milestone 1)

Minimal flow-matching image generator with a DiT backbone. Goal: overfit 10 images
and compare **x / ε / v prediction**.

| File | What |
|---|---|
| `dit.py` | small DiT: patchify → transformer blocks with adaLN-Zero (time-conditioned) → unpatchify |
| `flow.py` | flow-matching loss + ODE sampler; network output can mean x, ε or v |
| `train.py` | overfits the first N CIFAR-10 images, saves sample grids + curves |
| `sample.py` | generate images from a saved checkpoint |
| `utils.py` | data loading, grid saving, nearest-train-image MSE |

## Convention

`z_t = t·x + (1−t)·ε`, t=0 is noise, t=1 is the image, velocity `v = x − ε`.
Any one of x, ε, v gives the other two (formulas at the top of `flow.py`), so:

- `--pred` = what the network outputs
- `--loss-space` = where the MSE is computed (default: same as `--pred`)
- the sampler always converts the output to v and integrates `dz/dt = v` from t=0 to 1.

## Run

```
python train.py --pred v                    # v-pred, v-loss
python train.py --pred x                    # x-pred, x-loss
python train.py --pred eps                  # ε-pred, ε-loss
python train.py --pred x --loss-space v     # JiT setting: x-pred, v-loss
python sample.py runs/v-pred_v-loss/model.pt --method heun
```

CIFAR-10 is downloaded to `~/fastdata/cifar10` on the first run (~170 MB; on lab01 that is the
fast local disk, not backed up, re-downloadable). Change with `--data-root`.
Each run writes to `runs/<pred>-pred_<loss>-loss/`: `train_images.png`, `samples_<step>.png`,
`curves.png` (loss + MSE of samples to the nearest training image), `model.pt`, `log.json`.

Overfitting works when the samples look like copies of the training images and the
nearest-train MSE goes toward 0.
