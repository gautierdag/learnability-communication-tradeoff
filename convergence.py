from collections import defaultdict

from som import SelfOrganisingMap, NUM_CHIPS, sample_range
from typing import List, Dict, Tuple
from scipy.stats import ttest_1samp
import argparse
import numpy as np
import os
import sys
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def get_acc(p_t_s: np.ndarray, lid: int, som: SelfOrganisingMap, language: pd.DataFrame):
    correct = 0
    ts = np.argmax(p_t_s, axis=1)
    for cid, chip in language.groupby("chip"):
        if ts[cid - 1] == som.word_map[lid].inverse[chip['word'].mode()[0]]:
            correct += 1
    return correct / NUM_CHIPS


def evaluate_convergence(
        som: SelfOrganisingMap,
        data: pd.DataFrame,
        num_samples: int,
        prefix: str,
        language_ids: List[int] = None
) -> Dict[int, List[float]]:
    p_t_s = {}

    if language_ids is None:
        lids = range(1, som.term_data.nunique()['language'] + 1)
    else:
        lids = language_ids
    for lid in lids:
        try:
            # Should have shape (K,330,W) where K is the number of averaging runs
            p_t_s[lid] = np.load(os.path.join(prefix, f'{lid}/{num_samples}_pt_s_all.npy'))
        except OSError:
            print(os.path.join(prefix, f'{lid}/{num_samples}_pt_s.npy'))
            print(f'Error: no p(t|s) file found for language {lid}', file=sys.stderr)
            exit(1)

    return evaluate_convergence_model(som, data, p_t_s=p_t_s)


def evaluate_convergence_model(
        som: SelfOrganisingMap,
        data: pd.DataFrame,
        p_t_s: Dict[int, np.ndarray] = None
) -> Dict[int, List[float]]:
    accs = defaultdict(list)
    for lid, language in data.groupby("language"):
        for p in p_t_s[lid]:
            accs[lid].append(get_acc(p, lid, som, language))
    return accs


def n_sample_converged(window: int,
                       threshold: float,
                       accuracies: Dict[int, Dict[int, List[float]]],
                       language_ids: List[int]) -> Dict[int, int]:
    """ Find the number of samples where the SOM has converged.

    Args:
        window: Sliding window for convergence calculation
        threshold: What p-value to accept for convergence
        accuracies: Accuracies for the language at sampling steps
        language_ids: IDs for languages to test
    """
    lid_conv = {lid: list(sorted(accuracies))[-1] for lid in language_ids}
    for lid in language_ids:
        for i in np.arange(len(accuracies) - window):
            w = np.array([acc[lid] for j, (n, acc) in enumerate(accuracies.items()) if i <= j < i + window])
            w = w.mean(1)
            error = w - w.mean()
            rmsd = np.sqrt(np.dot(error, error) / (window - 1))
            if rmsd < threshold:
                lid_conv[lid] = sample_range[i]
                break
    return lid_conv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate convergence of learning to WCS data")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_samples", type=int, default=None, nargs='+',
                        help="Number of samples after which to evaluate")
    parser.add_argument("--lids", type=int, default=None, nargs='*', help="IDs of languages on which to evaluate")
    parser.add_argument("--prefix", type=str, default='output/som', help="Prefix of path to SOM scores")
    parser.add_argument("--plot", default=False, action='store_true', help="Produces a plot of the results.")
    parser.add_argument("--window", default=20, type=int, help="Sliding window for checking convergence")
    parser.add_argument("--threshold", default=0.01, type=float, help="P-value threshold check for t-test.")

    args = parser.parse_args()

    som = SelfOrganisingMap()
    data = som.term_data
    if args.lids is not None:
        data = data[data["language"].isin(args.lids)]
    data = data[~pd.isna(data["word"])]

    results = pd.DataFrame(columns=['language', 'n_samples', 'accuracy'])
    ns = args.n_samples
    if ns is None:
        ns = sample_range

    accuracies = {}
    for n in ns:
        print(f'Evaluating on {n} samples.', file=sys.stderr)
        accs = evaluate_convergence(som, data, n, os.path.join(args.prefix, str(args.seed)), language_ids=args.lids)
        accuracies[n] = accs
        df_dict = {}
        for lid, a in accs.items():
            df_dict["language"] = [lid] * len(a)
            df_dict["accuracy"] = a
            df_dict["n_samples"] = [n] * len(a)

        results = results.append(
            pd.DataFrame.from_dict(df_dict),
            ignore_index=True
        )

    n_conv = n_sample_converged(args.window, args.threshold, accuracies, args.lids)

    if args.plot:
        plt.ylim((0, 1))
        sns.set_theme()
        sns.lineplot(x="n_samples", y="accuracy",
                     hue="language",
                     data=results)

        for lid, cn in n_conv.items():
            plt.axvline(cn, linestyle="--")

        plt.show()
