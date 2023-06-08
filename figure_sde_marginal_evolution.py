# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
#
# Create a figure showing the evolution of the parameters of the SDE
import math

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets import WSJ0_mix
from pl_model import DiffSepModel
from sdes.sdes import MixSDE, PriorMixSDE

matplotlib.rc("text", usetex=True)
matplotlib.rcParams[
    "text.latex.preamble"
] = r"""
\usepackage{amsmath}
\usepackage{amsfonts}
"""

if __name__ == "__main__":

    # fig parameters
    limit = 5  # number of samples to use in the average (from validation set)
    d_lambda_s = [2, 3, 4]  # value of 'gamma' in the paper to explore
    cm2in = 0.39
    width = 8.5 * cm2in
    height = width / 2.5

    sns.set_theme(context="paper", style="white", font_scale=0.5)

    # fixed parameters
    sigma_min = 0.05
    sigma_max = 0.5
    t = torch.linspace(0.01, 1.0, 200)
    pal = sns.color_palette(palette="colorblind", n_colors=len(d_lambda_s))

    ds = WSJ0_mix("./data/wsj0_mix", 2, 8000, "max", "val")

    # plot some of the theoretical quantities
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(width, height))

    for lidx, d_lambda in enumerate(d_lambda_s):

        error_0 = t.new_zeros(t.shape)
        error_T = t.new_zeros(t.shape)
        N = 0

        sde = MixSDE(
            ndim=2, d_lambda=d_lambda, sigma_min=sigma_min, sigma_max=sigma_max, N=30
        )
        for i, (mix, tgt) in tqdm(enumerate(ds)):

            if i >= limit:
                break

            tgt = tgt[None, ...]
            mix = mix[None, ...]

            (mix, tgt), *stats = DiffSepModel.normalize_batch((mix, tgt))

            x0 = tgt
            xT = torch.broadcast_to(0.5 * mix, (1, 2, mix.shape[-1]))

            mu, L = sde.marginal_prob(tgt, t, mix)

            error_0 += torch.sum((x0 - mu) ** 2, dim=(1, 2))
            error_T += torch.sum((xT - mu) ** 2, dim=(1, 2))
            N += x0.shape[1] * x0.shape[2]

            if i == 0:
                # those are always the same
                if L.ndim == 3:
                    covmat = L @ L
                else:
                    L2 = L.to(0)
                    covmat = torch.einsum("bcdt,bdet->bce", L2, L2).cpu() / (
                        L.shape[-1]
                    )
                mean_mat = sde._mean_mix_mat(t)

                # assumes input mixture power is 1.0
                if L.ndim == 4:
                    sigma_mix = sde._std_sigma_mix(mix)
                    sigma_mix = sigma_mix[:, 0, :] ** 2
                else:
                    sigma_mix = 1.0
                snr = 10.0 * torch.log10(
                    sigma_mix / (L[:, 0, 0] ** 2 + L[:, 0, 1] ** 2)
                ).mean(dim=-1)

                ax2.plot(
                    t,
                    mean_mat[:, 0, 0],
                    "-",
                    c=pal[lidx],
                    label=f"$\gamma={{{d_lambda}}}$",
                )
                ax2.plot(t, mean_mat[:, 0, 1], "--", c=pal[lidx])

                var = covmat[:, 0, 0]
                cov = covmat[:, 0, 1]

                ax3.plot(t, cov / var, c=pal[lidx], label=f"$\gamma={{{d_lambda}}}$")

        error_0 /= N
        error_T /= N

        ax1.semilogy(t, error_T, "-", c=pal[lidx], label=f"$\gamma={{{d_lambda}}}$")
        # ax1.semilogy(t, error_0, "--", c=pal[lidx])

    # style the plots
    ax1.set_title(r"$\mathbb{E}\|\boldsymbol{\mu}_t - \bar{\boldsymbol{s}}\|^2$")
    ax1.set_xlabel("Time $t$")

    ax2.set_title(r"Coefficients of $e^{-t \gamma \bar{\boldsymbol{P}}}$")
    ax2.set_xlabel("Time $t$")

    ax3.set_title("Pearson corr. coeff.")
    ax3.set_xlabel("Time $t$")
    ax3.legend()

    print(f"Mixture error: {error_T[-1]}")

    sns.despine(fig=fig)
    fig.tight_layout(pad=0.3, h_pad=0.75)
    fig.savefig("figures/paper_sde_marginal_evolution.pdf")
    plt.close(fig)
