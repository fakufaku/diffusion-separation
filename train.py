# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import copy
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm

import utils
from datasets import WSJ0_mix_Module, Valentini_Module
from pl_model import DiffSepModel

log = logging.getLogger(__name__)


def load_model(config):

    if "score_model" in config.model:
        model_type = "score_model"
        model_obj = DiffSepModel
    else:
        raise ValueError("config/model should have a score_model sub-config")

    load_pretrained = getattr(config, "load_pretrained", None)
    if load_pretrained is not None:
        ckpt_path = Path(to_absolute_path(load_pretrained))
        hparams_path = (
            ckpt_path.parents[1] / "hparams.yaml"
        )  # path when using lightning checkpoint
        hparams_path_alt = (
            ckpt_path.parents[0] / "hparams.yaml"
        )  # path when using calibration output checkpoint

        log.info(f"load pretrained:")
        log.info(f"  - {ckpt_path=}")

        if hparams_path_alt.exists():
            log.info(f"  - {hparams_path_alt=}")
            # this was produced by the calibration routing
            with open(hparams_path, "r") as f:
                conf = yaml.safe_load(f)
                config_seld_model = conf["config"]["model"][model_type]

            config.model.seld_model.update(config_seld_model)
            model = model_obj(config)

            state_dict = torch.load(str(ckpt_path))

            log.info("Load model state_dict")
            model.load_state_dict(state_dict, strict=True)

        elif hparams_path.exists():
            log.info(f"  - {hparams_path=}")
            # this is a checkpoint
            with open(hparams_path, "r") as f:
                conf = yaml.safe_load(f)
                config_seld_model = conf["config"]["model"][model_type]

            config.model.seld_model.update(config_seld_model)

            log.info("Load model from lightning checkpoint")
            model = model_obj.load_from_checkpoint(
                ckpt_path, strict=True, config=config
            )

        else:
            raise ValueError(
                f"Could not find the hparams.yaml file for checkpoint {ckpt_path}"
            )

    else:
        model = model_obj(config)

    return model, (load_pretrained is not None)


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    if utils.ddp.is_rank_zero():
        exp_name = HydraConfig().get().run.dir
        log.info(f"Start experiment: {exp_name}")
    else:
        # when using DDP, if not rank zero, we are already in the run dir
        os.chdir(hydra.utils.get_original_cwd())

    # seed all RNGs for deterministic behavior
    pl.seed_everything(cfg.seed)

    torch.autograd.set_detect_anomaly(True)

    callbacks = []
    # Use a fancy progress bar
    callbacks.append(pl.callbacks.RichProgressBar())
    # configure checkpointing to save all models
    # save_top_k == -1  <-- saves all models
    val_loss_name = f"{cfg.model.main_val_loss}"
    loss_name = val_loss_name.split("/")[-1]  # avoid "/" in filenames
    modelcheckpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=val_loss_name,
        save_top_k=20,
        mode=cfg.model.main_val_loss_mode,
        filename="".join(
            ["epoch-{epoch:03d}_", loss_name, "-{", val_loss_name, ":.3f}"]
        ),
        auto_insert_metric_name=False,
    )
    callbacks.append(modelcheckpoint_callback)

    # the data module
    print("Using the DCASE2020 SELD original dataset")
    log.info("create datalogger")

    if cfg.name == "enhancement":
        dm = Valentini_Module(cfg)
    else:
        dm = WSJ0_mix_Module(cfg)

    # init model
    log.info("Create new model")
    model, is_pretrained = load_model(cfg)

    # create a logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="")

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints,
    # logs, and more)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=tb_logger)

    if cfg.train:
        log.info("start training")
        ckpt_path = getattr(cfg, "resume_from_checkpoint", None)
        if ckpt_path is None:
            trainer.fit(model, dm)
        else:
            trainer.fit(model, dm, ckpt_path=to_absolute_path(ckpt_path))

    if cfg.test:
        try:
            log.info("start testing")
            trainer.test(model, dm, ckpt_path="best")
        except pl.utilities.exceptions.MisconfigurationException:
            log.info(
                "test with current model value because no best model path is available"
            )
            trainer.validate(model, dm)
            trainer.test(model, dm)


if __name__ == "__main__":
    main()
