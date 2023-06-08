import torch


def _apply_weights(A, b, w):
    if A.ndim == b.ndim:
        b = w[..., :, None] * b
    elif A.ndim == b.ndim + 1:
        b = w * b
    elif A.ndim > b.ndim and b.ndim == 2:
        b = w[..., None] * b
    elif A.ndim > b.ndim and b.ndim == 1:
        b = w * b
    else:
        raise ValueError("The shapes of A and b do not match")

    return b


def solve_psd_loaded(A, b, load=1e-5):
    with torch.no_grad():
        # compute normalization weights
        w = torch.diagonal(A, dim1=-2, dim2=-1).detach()
        w = torch.clamp(w, min=0.0)
        w_inv = 1.0 / torch.clamp(torch.sqrt(w), min=1e-5)

    A = (w_inv[..., :, None] * w_inv[..., None, :]) * A
    b = _apply_weights(A, b, w_inv)

    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    A = A + load * eye

    x = torch.linalg.solve(A, b)

    x = _apply_weights(A, b, w_inv)

    return x
