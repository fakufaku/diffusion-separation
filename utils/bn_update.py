import torch
import tqdm


def get_batch_size(x):
    if isinstance(x, torch.Tensor):
        return x.size(0)
    elif isinstance(x, dict):
        if len(x) == 0:
            raise ValueError("Empty batch")
        key = next(iter(x.keys()))
        return get_batch_size(x[key])
    elif isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty batch")
        return get_batch_size(x[0])
    else:
        raise ValueError(
            f"Don't know how to get batch size of object of type {type(x)}"
        )


def to(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to(d, device) for d in data]
    elif isinstance(data, tuple):
        return tuple([to(d, device) for d in data])
    elif isinstance(data, dict):
        return {key: to(val, device) for key, val in data.items()}
    else:
        raise ValueError(f"Don't know how to get process object of type {type(data)}")


def bn_update(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    if not _check_bn(model):
        return

    model_device = model.device

    if device is not None:
        model = model.to(device)

    was_training = model.training
    model.train()
    momenta = {}
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input in tqdm.tqdm(loader, "updating batch-norm"):

        if isinstance(input, (list, tuple)):
            input = input[0]

        b = get_batch_size(input)

        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum

        if device is not None:
            input = to(input, device)

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)

    model = model.to(model_device)
    return model


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
