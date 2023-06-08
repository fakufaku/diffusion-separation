from typing import Dict, List, Optional, Tuple, Union


def import_name(name: str):

    parts = name.split(".")

    if len(parts) == 1:
        return __import__(parts[0])

    module_name = ".".join(parts[:-1])
    target_name = parts[-1]

    module = __import__(module_name, fromlist=(target_name,))
    if hasattr(module, target_name):
        return getattr(module, target_name)
    else:
        raise ImportError(f"Could not import {target_name} from {module_name}")


def module_from_config(
    name: str, args: Optional[Union[Tuple, List]] = None, kwargs: Optional[Dict] = None
):
    """
    Get a model by its name
    Parameters
    ----------
    name: str
        Name of the model class
    kwargs: dict
        A dict containing all the arguments to the model
    """

    if args is None:
        args = tuple()

    if kwargs is None:
        kwargs = {}

    obj = import_name(name)

    return obj(*args, **kwargs)


def run_configured_func(config, *args, **kwargs):
    config = dict(config)

    if not isinstance(config, dict):
        raise TypeError(
            "The argument of a ConfigFunc should be a dictionary containing "
            "the keyword arguments for the function to call"
        )
    try:
        func = import_name(config.pop("_target_"))
    except KeyError:
        raise ValueError(
            "The arguments of a ConfigFunc should "
            "contain '_target_' with the function name"
        )

    for kw in config:
        if kw in kwargs:
            raise ValueError("Repeated keyword arg")
    for kw in kwargs:
        if kw in config:
            raise ValueError("Repeated keyword arg")

    kwargs.update(config)

    return func(*args, **kwargs)
