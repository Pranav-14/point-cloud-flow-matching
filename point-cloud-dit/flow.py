"""Flow matching with a switchable prediction target: x, eps or v.

Convention (rectified flow, same as JiT):
    z_t = t * x + (1 - t) * eps        x = clean data, eps ~ N(0, I)
    t = 0 -> pure noise,  t = 1 -> clean data
    velocity  v = dz_t/dt = x - eps

Because z_t and t are known, any ONE of (x, eps, v) gives the other two:
    from x:    eps = (z - t*x) / (1 - t)       v = (x - z) / (1 - t)
    from eps:  x   = (z - (1-t)*eps) / t       v = (z - eps) / t
    from v:    x   = z + (1 - t)*v             eps = z - t*v

So the network can output any of the three ("pred"), and the loss can be
computed in any of the three spaces ("loss_space"). The sampler always
needs v, so the output is converted to v before each ODE step.
The divisions blow up near t=1 (x-pred) or t=0 (eps-pred); like JiT we
clamp the denominator at t_eps.

Works for any data shape (B, ...), here point clouds (B, N, 2).
"""
import torch

PRED_TYPES = ("x", "eps", "v")


def _bcast(t, like):
    return t.view(-1, *[1] * (like.dim() - 1))


def interpolate(x, eps, t):
    t = _bcast(t, x)
    return t * x + (1 - t) * eps


def target(space, x, eps):
    return {"x": x, "eps": eps, "v": x - eps}[space]


def convert(out, pred, to, z, t, t_eps=0.05):
    """Convert network output `out` (of type `pred`) into type `to`."""
    if pred == to:
        return out
    t = _bcast(t, z)
    if pred == "x":
        x = out
        eps = (z - t * x) / (1 - t).clamp(min=t_eps)
    elif pred == "eps":
        eps = out
        x = (z - (1 - t) * eps) / t.clamp(min=t_eps)
    elif pred == "v":
        x = z + (1 - t) * out
        eps = z - t * out
    else:
        raise ValueError(pred)
    return target(to, x, eps)


class FlowMatching:
    def __init__(self, pred="v", loss_space="v", t_sampler="uniform", t_eps=0.05, ln_mean=-0.8, ln_std=0.8):
        assert pred in PRED_TYPES and loss_space in PRED_TYPES
        self.pred = pred
        self.loss_space = loss_space
        self.t_sampler = t_sampler
        self.t_eps = t_eps
        self.ln_mean, self.ln_std = ln_mean, ln_std  # logit-normal params (JiT defaults)

    def sample_t(self, n, device):
        if self.t_sampler == "uniform":
            return torch.rand(n, device=device)
        if self.t_sampler == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=device) * self.ln_std + self.ln_mean)
        raise ValueError(self.t_sampler)

    def loss(self, model, x):
        eps = torch.randn_like(x)
        t = self.sample_t(x.shape[0], x.device)
        z = interpolate(x, eps, t)
        out = model(z, t)
        pred = convert(out, self.pred, self.loss_space, z, t, self.t_eps)
        return ((pred - target(self.loss_space, x, eps)) ** 2).mean()

    def velocity(self, model, z, t):
        t = t.expand(z.shape[0]) if t.dim() == 0 else t
        return convert(model(z, t), self.pred, "v", z, t, self.t_eps)

    @torch.no_grad()
    def sample(self, model, shape, steps=50, method="euler", device="cpu", generator=None):
        """Integrate dz/dt = v from t=0 (noise) to t=1 (data)."""
        z = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(0, 1, steps + 1, device=device)
        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            dt = t_next - t
            v = self.velocity(model, z, t)
            if method == "heun" and i < steps - 1:  # last step stays Euler (t=1 would divide by ~0)
                z_euler = z + dt * v
                v_next = self.velocity(model, z_euler, t_next)
                z = z + dt * 0.5 * (v + v_next)
            else:
                z = z + dt * v
        return z
