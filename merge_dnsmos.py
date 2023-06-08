# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluate_mp import summarize

fieldnames = [
    "",
    "filename",
    "len_in_sec",
    "sr",
    "num_hops",
    "OVRL_raw",
    "SIG_raw",
    "BAK_raw",
    "OVRL",
    "SIG",
    "BAK",
]
types = {
    "filename": Path,
    "len_in_sec": float,
    "sr": int,
    "num_hops": int,
    "OVRL_raw": float,
    "SIG_raw": float,
    "BAK_raw": float,
    "OVRL": float,
    "SIG": float,
    "BAK": float,
}


def get_results_filepath(exp_path, split):
    filepath = exp_path / f"{split}.json"
    if not filepath.exists():
        return False
    else:
        return filepath


def get_dnsmos_filepath(exp_path, split):
    filepath = exp_path / f"{split}_dnsmos.csv"

    if not filepath.exists():
        return False
    else:
        return filepath


def parse_dnsmos_csv(filepath):

    dnsmos = defaultdict(lambda: {})

    with open(filepath, newline="") as csvfile:
        dnsmos_reader = csv.reader(csvfile, delimiter=",")
        for idx, row in enumerate(dnsmos_reader):
            if idx == 0:
                # check that this is a valid DNSMOS output file
                for f1, f2 in zip(row, fieldnames):
                    if f1 != f2:
                        raise ValueError(
                            f"There might be an error in the DNSMOS file ({f1} != {f2})"
                        )
            else:
                sample_idx, channel_idx = Path(row[1]).stem.split(".")
                sample_idx = int(sample_idx)
                channel_idx = int(channel_idx[3:])

                dnsmos_res = {}
                for key, val in zip(fieldnames[2:], row[2:]):
                    dnsmos_res[key] = types[key](val)

                dnsmos[sample_idx][channel_idx] = dnsmos_res

    if len(dnsmos) == 0:
        raise ValueError("Empty DNSMOS file")

    # run some checks on the dictionary
    num_chan = len(dnsmos[sample_idx])
    errors = {}
    for sample_idx, res in dnsmos.items():
        if num_chan != len(res):
            errors[sample_idx] = len(res)
    if len(errors) > 0:
        print(f"Found {len(errors)} errors")
        for sample_idx, num_el in errors.items():
            print(f"  - sample {sample_idx} has only {num_el} channels")

    # convert to desired output format
    dnsmos_output = {}
    for sample_idx, res in dnsmos.items():
        dnsmos_output[sample_idx] = {}
        for key in fieldnames[2:]:
            dnsmos_output[sample_idx][key] = []
            for idx in range(num_chan):
                dnsmos_output[sample_idx][key].append(dnsmos[sample_idx][idx][key])

    return dnsmos_output


def get_results_file(filepath):
    with open(filepath, "r") as f:
        res = json.load(f)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge DNSMOS evaluation results into main result file"
    )
    parser.add_argument("results_path", type=Path, help="Path to result folder")
    parser.add_argument(
        "--overwrite-results", action="store_true", help="Path to result folder"
    )
    args = parser.parse_args()

    for split in ["val", "test", "libri-clean", "libri-noisy"]:
        if not (results_path := get_results_filepath(args.results_path, split)):
            print(f"Seems evaluate.py has not been run for {split}. Skip.")
            continue
        else:
            print(f"{split}: found results file")

        if not (dnsmos_path := get_dnsmos_filepath(args.results_path, split)):
            print(f"Seems DNSMOS evaluation has not been run for {split}. Skip.")
            continue
        else:
            print(f"{split}: found DNSMOS file")

        dnsmos = parse_dnsmos_csv(dnsmos_path)
        results = get_results_file(results_path)

        for idx, metrics in results.items():
            idx = int(idx)
            if idx not in dnsmos:
                breakpoint()
                raise ValueError(f"Sample {idx} not found in DNSMOS file")
            metrics.update(dnsmos[idx])

        summary = summarize(results, ignore_inf=False)

        if args.overwrite_results:
            output_results = results_path
            output_summary = args.results_path / f"{split}_summary.json"
        else:
            output_results = args.results_path / f"{split}_with_dnsmos.json"
            output_summary = args.results_path / f"{split}_summary_with_dnsmos.json"

        with open(output_results, "w") as f:
            json.dump(results, f, indent=2)
        with open(output_summary, "w") as f:
            json.dump(summary, f, indent=2)

        print(summary)
