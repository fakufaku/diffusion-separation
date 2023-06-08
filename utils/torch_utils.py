import torch


def count_parameters(module):
    param_count = 0
    for p in module.parameters():
        param_count += p.numel()
    return param_count


def to_device(data, device="cpu", to_numpy=False):
    """recursively transfers tensors to cpu"""
    if to_numpy and device != "cpu":
        raise ValueError("to_numpy and device=gpu is not compatible")

    if isinstance(data, list):
        return [to_device(d, device, to_numpy) for d in data]
    elif isinstance(data, dict):
        outdict = {}
        for key, val in data.items():
            outdict[key] = to_device(val, device, to_numpy)
        return outdict
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
        if to_numpy:
            data = data.numpy()
        return data
    else:
        return data
