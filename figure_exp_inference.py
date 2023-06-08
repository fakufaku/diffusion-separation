# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

matplotlib.rc("text", usetex=True)
matplotlib.rcParams[
    "text.latex.preamble"
] = r"""
\usepackage{amsmath}
\usepackage{amsfonts}
"""


def parse_name(name):
    fields = name.split("_")
    ret = dict()
    for i in range(1, 5):
        key, val = fields[-i].split("-")
        print(key, val)
        if key == "N" or key == "corrstep":
            ret[key] = int(val)
        elif key == "snr":
            ret[key] = float(val)
        elif key == "denoise":
            ret[key] = bool(val)
        else:
            raise ValueError(f"Unexpected key {key}")

    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create figure from experiment data")
    parser.add_argument("path_base", type=Path, help="base path of experiment data")
    parser.add_argument("key", type=str, help="key to use as x axis")
    args = parser.parse_args()

    pesq = []
    sisdr = []

    for split in ["val", "test"]:
        exp_folder = args.path_base.parent
        base_name = args.path_base.name
        for subfold in exp_folder.rglob(f"{base_name}*"):
            inf_args = parse_name(subfold.name)

            summary = subfold / f"{split}_summary.json"

            if not summary.exists():
                continue

            with open(summary, "r") as f:
                data = json.load(f)

            pesq.append((inf_args[args.key], data["pesq"]))
            sisdr.append((inf_args[args.key], data["si_sdr"]))

    pesq = sorted(pesq, key=lambda x: x[0])
    sisdr = sorted(sisdr, key=lambda x: x[0])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot([a[0] for a in pesq], [a[1] for a in pesq])
    ax1.set_ylabel("pesq")
    ax1.set_xlabel(args.key)

    ax2.plot([a[0] for a in sisdr], [a[1] for a in sisdr])
    ax2.set_ylabel("si-sdr (dB)")
    ax2.set_xlabel(args.key)

    fig.tight_layout()
    fig.savefig(f"figures/paper_exp_inference_{args.key}.pdf")
