import abc

import torch
from utils.registry import Registry

from . import sdes

CorrectorRegistry = Registry("Corrector")


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the corrector.
        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes
        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@CorrectorRegistry.register(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        self.score_fn = score_fn
        self.n_steps = n_steps
        self.snr = snr

    def update_fn(self, x, t, *args, **kwargs):
        target_snr = self.snr
        pad_dim = (...,) + (None,) * (x.ndim - t.ndim)
        for _ in range(self.n_steps):
            grad = self.score_fn(x, t, *args)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = ((target_snr * noise_norm / grad_norm) ** 2 * 2).unsqueeze(0)
            x_mean = x + step_size[pad_dim] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[pad_dim]

        return x, x_mean


@CorrectorRegistry.register(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, (sdes.MixSDE)):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, x, t, *args, **kwargs):
        n_steps = self.n_steps
        target_snr = self.snr
        x_mean = x
        std = self.sde.marginal_prob(x, t, *args)[1]

        if std.ndim > 1:
            # this is a sqrt covariance matrix, compute standard deviation of data
            std = (std @ std)[:, 0, :].sum(dim=-1, keepdim=True).sqrt()
            std = std[(...,) + (None,) * (x.ndim - std.ndim)]

        for _ in range(n_steps):
            grad = self.score_fn(x, t, *args)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2
            x_mean = x + step_size * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)

        return x, x_mean


@CorrectorRegistry.register(name="ald2")
class AnnealedLangevinDynamics2(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, (sdes.MixSDE, sdes.PriorMixSDE)):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, x, t, *args, **kwargs):
        n_steps = self.n_steps
        target_snr = self.snr
        x_mean = x
        L = self.sde.marginal_prob(x, t, *args)[1]

        for _ in range(n_steps):
            grad = self.score_fn(x, t, *args)
            noise = torch.randn_like(x)

            step_size = 2 * target_snr ** 2
            # replace L @ L @ grad by the following
            grad = self.sde.mult_std(L, grad)
            grad = self.sde.mult_std(L, grad)
            x_mean = x + step_size * grad

            step_size_2 = 2 * target_snr * L
            x = x_mean + self.sde.mult_std(step_size_2, noise)

        return x, x_mean


@CorrectorRegistry.register(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, *args, **kwargs):
        self.snr = 0
        self.n_steps = 0
        pass

    def update_fn(self, x, t, *args):
        return (x,)
