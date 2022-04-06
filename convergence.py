from collections import defaultdict
from tqdm import tqdm
import pickle

from som import SelfOrganisingMap, NUM_CHIPS
from typing import List, Dict
from numpy.lib.stride_tricks import sliding_window_view

from pathlib import Path
import argparse
import numpy as np
import glob

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def get_accuracy(
    p_t_s: np.ndarray,
    source: np.ndarray = None,
):
    correct = 0
    ts = np.argmax(p_t_s, axis=1)
    source_modes = np.argmax(source, axis=1)
    for cid in range(NUM_CHIPS):
        if ts[cid] == source_modes[cid]:
            correct += 1
    return correct / NUM_CHIPS


def evaluate_convergence(
    file: str,
) -> Dict[int, List[float]]:
    # Should have shape (K,330,W) where K is the number of averaging runs
    p_t_s = np.load(file)
    # Should have shape (330,W)
    source = np.load(Path(file).parent / "source.npy")
    K = p_t_s.shape[0]
    accuracies = []
    for k in range(K):
        accuracies.append(get_accuracy(p_t_s[k], source))
    return accuracies


def n_sample_converged(
    window: int,
    threshold: float,
    accuracies: Dict[str, Dict[int, List[float]]],
) -> Dict[int, int]:
    """Find the number of samples where the SOM has converged.

    Args:
        window: Sliding window for convergence calculation
        threshold: What p-value to accept for convergence
        accuracies: Accuracies dictionary {lid: {iteration_number: accuracy}}
    """
    lid_conv = {}
    for lid, accuracy_dict in tqdm(accuracies.items()):
        sorted_iterations = list(sorted(accuracy_dict))

        w = []
        for i in sorted_iterations:
            w.append(accuracy_dict[i])
        # W is a NxK matrix
        w = np.array(w)

        # using a sliding window in the N axis, V is (N-W)xKxW
        v = sliding_window_view(w, window, axis=0)

        # apply avg pooling on window
        v = v.mean(-1)

        # calculate rmsd
        error = v - v.mean(1)[:, None]
        rmsd = np.sqrt((error ** 2).sum(1) / (window - 1))  # @Balint Question why -1?

        idx = np.argmax(rmsd < threshold)
        if idx == 0 and not (rmsd < threshold).any():
            lid_conv[lid] = sorted_iterations[-1]
        else:
            lid_conv[lid] = sorted_iterations[idx]
    return lid_conv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate convergence of learning to WCS data"
    )
    parser.add_argument(
        "--prefix", type=str, default="lc_data", help="Prefix of path to SOM scores"
    )
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Produces a plot of the results.",
    )
    parser.add_argument(
        "--window", default=20, type=int, help="Sliding window for checking convergence"
    )
    parser.add_argument(
        "--threshold",
        default=0.01,
        type=float,
        help="P-value threshold check for t-test.",
    )

    args = parser.parse_args()

    som = SelfOrganisingMap()

    files = glob.glob("{args.prefix}/*/*/*_pt_s_all.npy")
    print(f"Found {len(files)} files.")

    results = pd.DataFrame(columns=["language", "n_samples", "accuracy"])

    accuracies = defaultdict(dict)
    lids = []
    accuracy_df = []
    n_samples = []

    for file in tqdm(files):
        # print(f"Evaluating on {n} samples.", file=sys.stderr)
        run_type = file.split("/")[-3]
        if run_type == "worst_qs":
            continue
        accs = evaluate_convergence(file)
        n = int(Path(file).stem.split("_")[0])
        lid = run_type + "_" + file.split("/")[-2]
        accuracies[lid][n] = accs

        accuracy_df += accs
        lids += [lid] * len(accs)
        n_samples += [n] * len(accs)

    pickle.dump(dict(accuracies), open("accuracies.pkl", "wb"))
    n_conv = n_sample_converged(args.window, args.threshold, dict(accuracies))

    print(n_conv)
    pickle.dump(dict(n_conv), open("n_conv.pkl", "wb"))

    if args.plot:
        results = pd.DataFrame(
            {"n_samples": n_samples, "language": lids, "accuracy": accuracy_df}
        )

        plt.ylim((0, 1))
        sns.set_theme()
        sns.lineplot(x="n_samples", y="accuracy", hue="language", data=results)

        for lid, cn in n_conv.items():
            plt.axvline(cn, linestyle="--")

        plt.show()
