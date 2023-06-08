import errno
import os
from pathlib import Path

import pytorch_lightning as pl


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def monkey_patch_add_best_symlink(callback, name="best-model"):
    """
    Patches the _save_topk_checkpoint method of ModelCheckpoint to also
    make a symlink of the best model available
    """

    assert isinstance(callback, pl.callbacks.ModelCheckpoint)

    original_func = pl.callbacks.ModelCheckpoint._save_topk_checkpoint

    def new_func(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # run original function
        original_func(self, trainer, pl_module)

        if self.best_model_path is not None:
            # make a symlink of the best checkpoint, if it exists
            src = Path(self.best_model_path).absolute()
            if src.exists():
                dst = (src.parent / name).with_suffix(src.suffix)
                symlink_force(src, dst)

    # replace the original method by the patched one
    callback._save_topk_checkpoint = new_func.__get__(
        callback, pl.callbacks.ModelCheckpoint
    )
