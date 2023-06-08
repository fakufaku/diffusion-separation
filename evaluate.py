# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import fast_bss_eval
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import yaml
from omegaconf import OmegaConf
from pesq import pesq
from pystoi import stoi

# from sdes.sdes import MixSDE
from datasets import NoisyDataset, WSJ0_mix
from pl_model import DiffSepModel

output_dir_base = Path("results")


def save_fig(x_result, intmet, target, fig_out_dir, n_fig=6, vmin=-75, vmax=0):
    # back to cpu
    x_result = x_result.cpu()
    target = target.cpu()

    # Save figure of evolution
    fig, axes = plt.subplots(2, n_fig + 1, figsize=(20, 4))

    steps = np.round(np.linspace(0, 1, n_fig) * (len(intmet) - 1)).astype(np.int64)

    for idx, step in enumerate(steps):
        arr = intmet[step][0].cpu().numpy()
        for i in range(2):
            im = axes[i, idx].specgram(arr[0, i], vmin=vmin, vmax=vmax)
            axes[i, idx].set_xticks([])
            axes[i, idx].set_yticks([])
            if i == 0:
                axes[i, idx].set_title(
                    f"t={(len(intmet) - 1 - step) / (len(intmet) - 1):.2f}"
                )
    for i in range(2):
        tgt = target[0, i] + np.random.randn(*target[0, i].shape) * 1e-10
        *_, im = axes[i, -1].specgram(tgt, vmin=vmin, vmax=vmax)
        axes[i, -1].set_xticks([])
        axes[i, -1].set_yticks([])
        if i == 0:
            axes[i, -1].set_title("clean")
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(fig_out_dir / f"evo_{batch_idx:03d}.pdf")
    plt.close(fig)


def save_samples(mix, x_result, target, wav_out_dir, fs):
    # save samples
    all_wav = torch.cat(
        (mix[0].cpu(), x_result[0, :].cpu(), target[0].cpu()),
        dim=0,
    )
    all_wav = all_wav.cpu()
    all_wav = all_wav[:, None, :]

    max_val = abs(all_wav).max()
    all_wav *= 0.95 / max_val

    torchaudio.save(
        str(wav_out_dir / f"{batch_idx:03d}_mix.wav"),
        all_wav[0],
        fs,
    )
    torchaudio.save(
        str(wav_out_dir / f"{batch_idx:03d}_enh0.wav"),
        all_wav[1],
        fs,
    )
    torchaudio.save(
        str(wav_out_dir / f"{batch_idx:03d}_enh1.wav"),
        all_wav[2],
        fs,
    )
    torchaudio.save(
        str(wav_out_dir / f"{batch_idx:03d}_tgt0.wav"),
        all_wav[3],
        fs,
    )
    torchaudio.save(
        str(wav_out_dir / f"{batch_idx:03d}_tgt1.wav"),
        all_wav[4],
        fs,
    )


def compute_metrics(ref, est, fs, pesq_mode="nb", stoi_extended=True, n_src=2):

    si_sdr, si_sir, si_sar, perm = fast_bss_eval.si_bss_eval_sources(
        ref,
        est,
        zero_mean=False,
        compute_permutation=True,
        clamp_db=100,
    )

    # order according to SIR
    est = est[:, perm[0], :]

    est = est.cpu().numpy()
    ref = ref.cpu().numpy()

    p_esq = []
    s_toi = []
    for src_idx in range(n_src):
        p_esq.append(pesq(fs, ref[0, src_idx], est[0, src_idx], pesq_mode))
        s_toi.append(stoi(ref[0, src_idx], est[0, src_idx], fs, extended=stoi_extended))

    return (
        si_sdr[..., :n_src],
        si_sir[..., :n_src],
        si_sar[..., :n_src],
        p_esq,
        s_toi,
        perm,
    )


def summarize(results):
    metrics = set()
    summary = defaultdict(lambda: 0)

    for res in results:
        for met, val in res.items():
            metrics.add(met)
            summary[met] += np.mean(val)
        summary["number"] += 1

    return si_sdr, si_sir, si_sar, p_esq, s_toi, perm


def summarize(results):
    metrics = set()
    summary = defaultdict(lambda: 0)

    for res in results:
        for met, val in res.items():
            metrics.add(met)
            summary[met] += np.mean(val)
        summary["number"] += 1

    for met in metrics:
        summary[met] = (summary[met] / summary["number"]).tolist()

    return dict(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation on validation or test dataset"
    )
    parser.add_argument("ckpt", type=Path, help="Path to checkpoint to use")
    parser.add_argument(
        "-o", "--output_dir", type=Path, default="results", help="The output folder"
    )
    parser.add_argument(
        "--enhance",
        default=False,
        action="store_true",
        help="Compute evaluation metris for speech enhancement",
    )
    parser.add_argument(
        "-d", "--device", default=0, help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "-w",
        "--dl-workers",
        type=int,
        help="Number of workers for the dataloader (default os.cpu_count())",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help=(
            "A tag name for the experiment. If not provided,"
            " the experiment and checkpoints name are used."
        ),
    )
    parser.add_argument(
        "-l", "--limit", type=int, help="Limit the number of samples to process"
    )
    parser.add_argument(
        "--save-n",
        type=int,
        help="Save a limited number of output samples (default: save all)",
    )
    parser.add_argument("--val", action="store_true", help="Run on validation dataset")
    parser.add_argument("--test", action="store_true", help="Run on test dataset")
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
        "--pesq-mode",
        type=str,
        choices=["nb", "wb"],
        default="nb",
        help="Mode for PESQ 'wb' or 'nb'",
    )
    parser.add_argument(
        "--stoi-no-extended", action="store_true", help="Disable extended mode for STOI"
    )
    parser.add_argument(
        "-s", "--schedule", type=str, help="Pick a different schedule for the inference"
    )

    args = parser.parse_args()

    if not (args.val or args.test):
        parser.error("No action requested, add --val or --test")

    device = args.device
    n_samples_limit = args.limit
    output_dir_base = args.output_dir
    if args.dl_workers is None:
        num_dl_workers = os.cpu_count()
    else:
        num_dl_workers = args.dl_workers

    # special case to get the original data
    no_proc_flag = str(args.ckpt) == "__no_proc__"

    if no_proc_flag:
        # load validation dataset
        datasets = dict()
        path = "data/wsj0_mix"
        n_spkr = 2
        cut = "max"
        if args.val:
            datasets["val"] = WSJ0_mix(path=path, n_spkr=n_spkr, cut=cut, split="val")
        if args.test:
            datasets["test"] = WSJ0_mix(path=path, n_spkr=n_spkr, cut=cut, split="test")

        if args.tag is None:
            output_dir = output_dir_base / "mix"
        else:
            output_dir = output_dir_base / args.tag

    else:
        # load the config file
        with open(args.ckpt.parents[1] / "hparams.yaml", "r") as f:
            hparams = yaml.safe_load(f)
        config = hparams["config"]

        datasets = dict()
        if args.enhance:
            config["datamodule"]["test"]["dataset"].pop("_target_")
            datasets["test"] = NoisyDataset(**config["datamodule"]["test"]["dataset"])
            n_src = 1

        # load validation dataset
        else:
            for split in ["val", "train"]:
                # remove the target because we don't use 'instantiate'
                config["datamodule"][split]["dataset"].pop("_target_")
                # check the location of the data
                data_path = Path(config["datamodule"][split]["dataset"]["path"])
                if not data_path.exists():
                    config["datamodule"][split]["dataset"]["path"] = "./data/wsj0_mix"

            if args.val:
                datasets["val"] = WSJ0_mix(**config["datamodule"]["val"]["dataset"])
            if args.test:
                datasets["test"] = WSJ0_mix(**config["datamodule"]["test"]["dataset"])
            n_src = 2

        # load model
        model = DiffSepModel.load_from_checkpoint(str(args.ckpt))

        # transfer to GPU
        model = model.to(device)
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
        tag_inf = f"N-{N}_snr-{snr}_corrstep-{corrector_steps}_denoise-{denoise}_schedule-{args.schedule}"

        # create folder name based on experiment and checkpoint
        exp_name = args.ckpt.parents[1].name
        ckpt_name = args.ckpt.stem
        if args.tag is None:
            output_dir = output_dir_base / f"{exp_name}_{ckpt_name}_{tag_inf}"
        else:
            output_dir = output_dir_base / f"{args.tag}_{tag_inf}"

    output_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = output_dir / "fig"
    wav_dir = output_dir / "wav"
    print(f"Created output folder {output_dir}")

    # wraps datasets into dataloaders
    dataloaders = {
        key: torch.utils.data.DataLoader(
            val,
            shuffle=False,
            num_workers=num_dl_workers,
            pin_memory=True,
            batch_size=1,
        )
        for key, val in datasets.items()
    }

    for split, dl in dataloaders.items():

        print(f"Processing {split}: {len(dl)} samples")

        results = []
        fs = dl.dataset.fs

        for batch_idx, (mix, target) in enumerate(dl):

            if n_samples_limit is not None and batch_idx >= n_samples_limit:
                break

            # decide if we want to save some sample and figure
            save_samples_fig = args.save_n is None or (batch_idx < args.save_n)

            mix = mix.to(device)
            target = target.to(device)
            length = target.shape[-1] / fs

            if no_proc_flag:
                x_result = torch.broadcast_to(mix, target.shape)
                nfe = 0
                intmet = None
                t_proc = 0.0
                save_samples_fig = False

            else:
                (mix, target), *__ = model.normalize_batch((mix, target))

                sampler = model.get_pc_sampler(
                    "reverse_diffusion",
                    "ald2",
                    mix,
                    N=N,
                    denoise=denoise,
                    intermediate=save_samples_fig,
                    corrector_steps=corrector_steps,
                    snr=snr,
                    schedule=args.schedule,
                )

                t_s = time.perf_counter()
                x_result, nfe, *others = sampler()
                t_proc = time.perf_counter() - t_s

                if len(others) > 0:
                    intmet = others[0]

            # compute the metrics before separation
            si_sdr, si_sir, si_sar, p_esq, s_toi, perm = compute_metrics(
                target,
                x_result,
                fs,
                pesq_mode=args.pesq_mode,
                stoi_extended=not args.stoi_no_extended,
                n_src=n_src,
            )

            # fix the permutation
            x_result = x_result[:, perm[0], :]

            results.append(
                {
                    "batch_idx": batch_idx,
                    "si_sdr": si_sdr.tolist()[:n_src],
                    "si_sir": si_sir.tolist()[:n_src],
                    "si_sar": si_sar.tolist()[:n_src],
                    "pesq": p_esq,
                    "stoi": s_toi,
                    "nfe": nfe,
                    "runtime": t_proc,
                    "len_s": length,
                }
            )

            print(f"{split}", end=" ")
            for met, val in results[-1].items():
                print(f"{met}={np.mean(val):.3f}", end=" ")
            print()

            if save_samples_fig:

                # fix permutations of intermediate results
                if intmet is not None:
                    for idx in range(len(intmet)):
                        xt, xt_mean = intmet[idx]
                        intmet[idx] = (xt[:, perm[0], :], xt_mean[:, perm[0], :])

                fig_out_dir = fig_dir / split
                fig_out_dir.mkdir(exist_ok=True, parents=True)
                wav_out_dir = wav_dir / split
                wav_out_dir.mkdir(exist_ok=True, parents=True)

                save_fig(
                    x_result,
                    intmet,
                    target,
                    fig_out_dir,
                    n_fig=6,
                    vmin=-75,
                    vmax=0,
                )

                save_samples(mix, x_result, target, wav_out_dir, fs)

        with open(output_dir / f"{split}.json", "w") as f:
            json.dump(results, f, indent=2)

        summary = summarize(results)
        with open(output_dir / f"{split}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
