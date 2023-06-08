# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import argparse
import random
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from scipy.io import wavfile
from scipy.signal import get_window, stft


def specgram_vignette(
    audio, crop_time=None, crop_freq=None, floor=1e-15, fs=16000, **kwargs
):

    f, t, Z = stft(audio, fs=fs, **kwargs)

    if crop_time is not None:
        select = np.logical_and(t >= crop_time[0], t <= crop_time[1])
        Z = Z[:, select]

    if crop_freq is not None:
        select = np.logical_and(f >= crop_freq[0], f <= crop_freq[1])
        Z = Z[select, :]

    A = 20.0 * np.log10(abs(Z) + floor)

    return A


def choose_samples(results, tiers=3, n_samples=1, key="si_sdr"):
    """bucket the results in `tiers` groups and pick one sample at random from each group"""

    # partition the into buckets
    values = [np.mean(r[key]) for r in results.values()]
    q = np.linspace(0, 100.0, tiers + 1)
    q = [0, 10, 90, 100]
    limits = np.percentile(values, q)

    # sort the samples into the buckets
    buckets = [[] for b in range(tiers)]
    val_buck = [[] for b in range(tiers)]
    for sample_idx, metrics in results.items():
        val = np.mean(metrics[key])
        for b in range(tiers):
            if limits[b] <= val <= limits[b + 1]:
                buckets[b].append(sample_idx)
                val_buck[b].append(val)

    # choose samples at random from each bucket
    representatives = [random.sample(bucket, k=n_samples) for bucket in buckets]

    return representatives


def read_specgram(folder, split, sample_idx, suffix, time_len, freq_max):

    sample_idx = int(sample_idx)

    audio = []
    for ch in range(2):
        path = folder / f"wav/{split}/{sample_idx:04d}.{suffix}{ch}.wav"
        if not path.exists():
            raise ValueError(f"Could not find file {path}")
        fs, a = wavfile.read(str(path))

        t0 = (a.shape[0] / fs - time_len) // 2
        t1 = t0 + time_len
        f0 = 0.0
        f1 = freq_max

        if a.dtype == np.int16:
            a = a / (2 ** 15)

        A = specgram_vignette(a, crop_time=[t0, t1], crop_freq=[f0, f1], fs=fs)

        audio.append(A)

    return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a figure with good and bad samples"
    )
    parser.add_argument("target_dir", type=Path, help="Path to clean files")
    parser.add_argument("enhance_dir", type=Path, help="Path to enhanced files")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "libri-clean", "libri-noisy"],
        help="Split of the dataset to use",
    )
    parser.add_argument(
        "--sort-by", default="si_sdr", help="metric to use to sort the samples"
    )

    random.seed(20171001)

    args = parser.parse_args()

    split = args.split

    tiers = 3
    n_samples = 1

    vmin = -70
    vmax = -13
    cm2in = 0.39
    figsize = (8.5 * cm2in, 3.3 * cm2in)

    sns.set_theme(context="paper", style="white", font_scale=0.5)

    # first we look at the metric
    results_path = args.enhance_dir / f"{split}.json"
    if not results_path.exists():
        raise ValueError(f"The result file does not exist for {split}")

    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    selected = choose_samples(results, tiers=3, n_samples=1, key=args.sort_by)

    fig, axes = plt.subplots(2 * n_samples, 2 * tiers, figsize=figsize)

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(n_samples, tiers)

    for b in range(tiers):
        for n in range(n_samples):

            inner = gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=outer[n, b], wspace=0.05, hspace=0.05
            )

            sidx = selected[b][n]

            metric = np.array(results[sidx][args.sort_by])
            if metric.ndim == 2:
                metric = metric[0]

            # specgram vignette of 5 seconds, up to 6 kHz at the middle of the segment
            t_len, freq_max = 3.0, 4000
            enh = read_specgram(args.enhance_dir, split, sidx, "enh", t_len, freq_max)
            tgt = read_specgram(args.target_dir, split, sidx, "tgt", t_len, freq_max)

            print("min", min([a.min() for a in enh + tgt]))
            print("max", max([a.max() for a in enh + tgt]))

            for i in range(2):
                ax = plt.Subplot(fig, inner[0, i])
                ax.set_title(f"{metric[i]:.2f} dB")
                ax.imshow(
                    enh[i],
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    aspect="auto",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if b == 0 and i == 0:
                    ax.set_ylabel("Separated")
                fig.add_subplot(ax)

                ax = plt.Subplot(fig, inner[1, i])
                ax.imshow(
                    tgt[i],
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    aspect="auto",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if b == 0 and i == 0:
                    ax.set_ylabel("Clean")
                fig.add_subplot(ax)

    outer.tight_layout(fig, pad=0.0, w_pad=1.5, h_pad=0.0)

    fig.savefig("figures/paper_samples.pdf", dpi=300)
    plt.close(fig)
