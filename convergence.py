from som import SelfOrganisingMap, NUM_CHIPS
from typing import List, Dict
import argparse
import numpy as np
import os
import sys
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_convergence(
    num_samples: int,
    prefix: str,
    language_ids: List[int] = None
) -> Dict[int, float]:
    p_t_s = {}
    som = SelfOrganisingMap()

    if language_ids is None:
        lids = range(1, som.term_data.nunique()['language']+1)
    else:
        lids = language_ids
    for lid in lids:
        try:
            p_t_s[lid] = np.load(os.path.join(prefix, f'{lid}/{num_samples}_pt_s.npy'))
        except OSError:
            print(os.path.join(prefix, f'{lid}/{num_samples}_pt_s.npy'))
            print(f'Error: no p(t|s) file found for language {lid}', file=sys.stderr)
            exit(1)
    
    return evaluate_convergence_model(som, p_t_s=p_t_s, language_ids=language_ids)


def evaluate_convergence_model(
    som: SelfOrganisingMap,
    p_t_s: np.ndarray = None,
    language_ids: List[int] = None
) -> Dict[int, float]:
    data = som.term_data

    if p_t_s is None:
        p_t_s = som.predict_t_s(language_ids=language_ids)

    if language_ids is not None:
        data = data[data["language"].isin(language_ids)]
        
    accs = {}
    for lid, language in data.groupby("language"):
        correct = 0
        ts = np.argmax(p_t_s[lid], axis=1)
        for cid, chip in language.groupby("chip"):
            if ts[cid-1] == som.word_map[lid].inverse[chip['word'].mode()[0]]:
                correct += 1
        accs[lid] = correct/NUM_CHIPS
    return accs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate convergence of learning to WCS data")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_samples", type=int, default=[50000], nargs='+', help="Number of samples after which to evaluate")
    parser.add_argument("--lids", type=int, default=None, nargs='*', help="IDs of languages on which to evaluate")
    parser.add_argument("--prefix", type=str, default='output/som', help="Prefix of path to SOM scores")
    parser.add_argument("--plot", default=False, action='store_true', help="Produces a plot of the results.")


    args = parser.parse_args()
    results = pd.DataFrame(columns=['language', 'n_samples', 'accuracy'])
    for n in args.n_samples:
        print(f'Evaluating on {n} samples.', file=sys.stderr)
        accs = evaluate_convergence(n, os.path.join(args.prefix, str(args.seed)), language_ids=args.lids)
        results = results.append(
            pd.DataFrame({'language': accs.keys(), 'accuracy': accs.values(), 'n_samples': [n] * len(accs)}), 
            ignore_index=True
        )
    if args.plot:
        plt.ylim((0, 1))
        sns.set_theme()
        sns.lineplot(x="n_samples", y="accuracy",
                hue="language",
                data=results)
        plt.show()
    print(results.to_csv(sep='\t', index=False))
