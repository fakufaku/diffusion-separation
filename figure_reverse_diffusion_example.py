# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
#
# The path to the checkpoint on line 32 should be changed to match
# the checkpoint to use
from pathlib import Path

import fast_bss_eval
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from sdes.sdes import MixSDE
from datasets import WSJ0_mix
from pl_model import DiffSepModel
from sdes import get_pc_sampler

if __name__ == "__main__":

    device = 0
    n_samples = 1

    n_fig = 5
    vmin = -40
    vmax = 0
    cm2in = 0.39
    figsize = (8.5 * cm2in, 3.5 * cm2in)

    sns.set_theme(context="paper", style="white", font_scale=0.5)

    # load model
    checkpoint = Path(
        "exp/default/2022-10-21_02-52-17_experiment-model-large-multigpu"
        "/checkpoints/epoch-644_si_sdr-10.965.ckpt"
    )
    model = DiffSepModel.load_from_checkpoint(str(checkpoint))

    # load validation dataset
    ds = WSJ0_mix("./data/wsj0_mix", 2, 8000, "max", "test")

    # transfer to GPU
    model = model.to(device)
    model.eval()

    for batch_idx, (mix, target) in enumerate(ds):

        if batch_idx >= n_samples:
            break

        mix = mix[None, ...].to(device)
        target = target[None, ...].to(device)
        length = target.shape[-1] / ds.fs

        (mix, target), *__ = model.normalize_batch((mix, target))

        model.sde.N = 30
        sampler = get_pc_sampler(
            "reverse_diffusion",
            "ald2",
            model.sde,
            model,
            mix,
            # fake_mix=fake_mix,
            eps=model.t_eps,
            denoise=True,
            intermediate=True,
            corrector_steps=1,
            snr=0.5,
        )
        x_result, ns, intmet = sampler()

        si_sdr, perm = fast_bss_eval.si_sdr(
            target, x_result, zero_mean=False, return_perm=True
        )
        x_result = x_result[:, perm[0], :]

        if intmet is not None:
            for idx in range(len(intmet)):
                xt, xt_mean = intmet[idx]
                intmet[idx] = (xt[:, perm[0], :], xt_mean[:, perm[0], :])
        print(f"{batch_idx:03d} SI-SDR {si_sdr.cpu().tolist()} ({length:.2f} s)")

        # back to cpu
        x_result = x_result.cpu()
        target = target.cpu()
        mix = mix.cpu()

        if intmet is not None:
            fig, axes = plt.subplots(2, n_fig, figsize=figsize)
            times = np.linspace(0, 1, n_fig)
            steps = np.round(times * (len(intmet) - 1)).astype(np.int64)

            for idx, step in enumerate(steps):
                arr = intmet[step][0].cpu().numpy()
                loc = n_fig - 1 - idx
                for i in range(2):
                    im = axes[i, loc].specgram(arr[0, i], vmin=vmin, vmax=vmax)
                    axes[i, loc].set_xticks([])
                    axes[i, loc].set_yticks([])

            fig.tight_layout(h_pad=0.2, w_pad=0.2, pad=0.1)

            sns.despine(fig=fig, left=True, bottom=True)
            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            # fig.colorbar(im, cax=cbar_ax)
            fig.savefig(f"figures/paper_reverse_diffusion_example_{batch_idx:03d}.pdf")
            plt.close(fig)
