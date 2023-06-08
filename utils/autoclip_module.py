import bisect
from typing import Tuple

import torch


def grad_norm(module: torch.nn.Module):
    """
    Helper function that computes the gradient norm
    """
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class FixedClipper:
    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def __call__(self, module: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            module.parameters(),
            max_norm=self.max_norm,
            norm_type=2,
            error_if_nonfinite=False,
        )
        return grad_norm, self.max_norm


class AutoClipper:
    """
    This class can be used to automatically adjust the threshold used
    to clip the gradient

    Parameters
    ----------
    p: int
        The percentile between 0 and 100 where we choose the threshold
        in the gradient history.
    """

    def __init__(self, p: float):
        self.autoclip_p = p / 100
        self.grad_norm_history = []

    def __call__(self, module: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the norm of the gradient of module, adjusts the clipping
        threshold, and apply the clipping to the weights of module.
        """

        # compute the gradient norm and insert in the list
        # in sorted order
        gnorm = grad_norm(module)
        bisect.insort(self.grad_norm_history, gnorm)

        # select the pth percentile
        index = int(self.autoclip_p * len(self.grad_norm_history))
        if index == len(self.grad_norm_history):
            index -= 1
        grad_clip_norm = self.grad_norm_history[index]

        # perform the clipping
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=grad_clip_norm)

        return gnorm, grad_clip_norm
