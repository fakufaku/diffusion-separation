# 2024 (c) LY Corporation
# Authors: Robin Scheibler
# MIT License
import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf

# from sdes.sdes import MixSDE
from pl_model import DiffSepModel

DEFAULT_MODEL = "fakufaku/diffsep"


def str_or_int(x):
    try:
        x = int(x)
    except ValueError:
        pass
    return x


def get_model(args):
    if not args.model.exists():
        # assume this is a HF model
        path = hf_hub_download(repo_id=str(args.model), filename="checkpoint.pt")
    else:
        path = args.model

    # load model
    model = DiffSepModel.load_from_checkpoint(str(path))

    # transfer to GPU
    model = model.to(args.device)
    model.eval()

    # prepare inference parameters
    sampler_kwargs = model.config.model.sampler
    N = sampler_kwargs.N if args.N is None else args.N
    corrector_steps = (
        sampler_kwargs.corrector_steps
        if args.corrector_steps is None
        else args.corrector_steps
    )
    snr = sampler_kwargs.snr if args.snr is None else args.snr
    denoise = args.denoise

    kwargs = {
        "N": N,
        "denoise": denoise,
        "intermediate": False,
        "corrector_steps": corrector_steps,
        "snr": snr,
        "schedule": args.schedule,
    }

    return model, kwargs


def scale_output(mix, sep):
    # project mix onto separated signal
    num = (mix * sep).sum(dim=-1, keepdim=True)
    denom = (sep * sep + 1e-10).sum(dim=-1, keepdim=True)
    alpha = num / denom
    return alpha * sep


def separate(mix, model, sampler_kwargs, device):
    mix = mix.to(device)
    mix = mix[None]  # add batch dim

    (mix_norm, _), *__ = model.normalize_batch((mix, None))

    sampler = model.get_pc_sampler(
        "reverse_diffusion",
        "ald2",
        mix_norm,
        **sampler_kwargs,
    )

    with torch.no_grad():
        sep, nfe, *_ = sampler()

    sep = scale_output(mix, sep)

    return sep.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separate all the wav files in a specified folder"
    )
    parser.add_argument("input_dir", type=Path, help="Path to the input folder")
    parser.add_argument("output_dir", type=Path, help="Path to the output folder")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to model or Huggingface model",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str_or_int,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument("-N", type=int, default=None, help="Number of steps")
    parser.add_argument(
        "--snr", type=float, default=None, help="Step size of corrector"
    )
    parser.add_argument(
        "--corrector-steps", type=int, default=None, help="Number of corrector steps"
    )
    parser.add_argument(
        "--denoise", type=bool, default=True, help="Use denoising in solver"
    )
    parser.add_argument(
        "-s", "--schedule", type=str, help="Pick a different schedule for the inference"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = "cpu"
        print("No CUDA, fall back to CPU")

    model, sampler_kwargs = get_model(args)
    model_sr = model.config.model.fs

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    elif args.output_dir.is_file():
        raise ValueError("Output directory is a file")

    for wavpath in tqdm(args.input_dir.glob("*.wav"), desc="Separating wav files"):
        waveform, sr = torchaudio.load(wavpath)

        if sr != model_sr:
            print(
                f"Skipping {wavpath.stem} due to mismatched sample rate. "
                f"This model expects {model_sr} Hz, but the file is {sr} Hz."
            )
        sep = separate(waveform, model, sampler_kwargs, args.device)
        for i in range(sep.shape[1]):
            spkr_dir = args.output_dir / f"s{i}"
            spkr_dir.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                spkr_dir / f"{wavpath.stem}.wav", sep[:, i, :], sr, format="wav"
            )
