from collections import defaultdict
from tqdm import tqdm
import pickle

from som import SelfOrganisingMap, NUM_CHIPS
from typing import List, Dict

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
    accuracies: Dict[str, Dict[int, List[float]]],
    patience: int = 50,
    threshold: float = 0.01,
) -> Dict[str, List[int]]:
    """Does simple Early Stopping to find where the SOM has converged for every language and every seed

    Args:
        patience: time to wait between checkpoint before deciding that the SOM has converged
        threshold: What minimum accuracy increase to consider as improvement (default, 1%)
        accuracies: Accuracies dictionary {lid: {iteration_number: accuracy}}
    """
    lid_conv = {}
    lids = list(sorted(accuracies))
    for lid in tqdm(lids):
        accuracy_dict = accuracies[lid]
        sorted_iterations = list(sorted(accuracy_dict))

        idxs = []
        K = len(accuracy_dict[1])
        for seed in range(K):
            best = 0
            best_iteration = 1
            waiting = 0
            for j in sorted_iterations:
                if best + threshold < accuracy_dict[j][seed]:
                    best = accuracy_dict[j][seed]
                    best_iteration = j
                    waiting = 0
                if waiting == patience:
                    break
                waiting += 1

            idxs.append(best_iteration)
        idxs = np.array(idxs)
        lid_conv[lid] = idxs
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
    args = parser.parse_args()

    som = SelfOrganisingMap()

    files = glob.glob(f"{args.prefix}/*/*/*_pt_s_all.npy")
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
    n_conv = n_sample_converged(dict(accuracies))
    pickle.dump(dict(n_conv), open("n_conv.pkl", "wb"))

    if args.plot:
        results = pd.DataFrame(
            {"n_samples": n_samples, "language": lids, "accuracy": accuracy_df}
        )
        plt.ylim((0, 1))
        sns.set_theme()
        sns.lineplot(x="n_samples", y="accuracy", hue="language", data=results)
        for lid, cn in n_conv.items():
            plt.axvline(np.median(cn), linestyle="--")
        plt.show()
