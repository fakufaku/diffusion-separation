import torch


def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": torch.nn.SILU,
    }

    return activation_funcs[act]()
