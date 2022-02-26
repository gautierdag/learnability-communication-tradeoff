import os.path
import pickle
import sys

import multiprocessing
import numpy as np

from ibhelpers import fit_ib
from som import SelfOrganisingMap


def fit_optimal_curve(args):
    lid = args[0]
    q = args[1].T
    num_cols = q.shape[1]
    if num_cols < 330:
        duplicate_vector = np.ones(num_cols, dtype=int)
        duplicate_vector[-1] = 331 - num_cols
        q = np.repeat(q, duplicate_vector, axis=1)
    q[:, 330 - duplicate_vector[-1]:] /= duplicate_vector[-1]
    q[np.isnan(q)] = 0
    # uniform dist for q va
    q[q.sum(1) == 0] = 1 / q.shape[0]

    q0 = np.eye(q.shape[1])
    focalbeta = 1
    betas = np.array([2 ** x for x in np.arange(4, -2, -0.5)])
    scores = fit_ib(q, q0, focalbeta, betas, verbose=1, divergence="entropy")[2]

    if len(args) < 3:
        pickle.dump(scores, open(f"results/learnability_scores/{lid}.p", "wb"))
    else:
        pickle.dump(scores, open(args[2], "wb"))
    return scores


if __name__ == '__main__':
    multiproc = False
    workers = 110

    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/learnability_scores"):
        os.mkdir("results/learnability_scores")

    # Create a SOM with arbitrary params to load data and calculate data distributions
    som = SelfOrganisingMap(1, 0.5, 1, 1)

    items = list(som.pts.items())

    if not multiproc:
        for params in items:
            if len(sys.argv) > 1 and params[0] != int(sys.argv[1]):
                continue
            print(f"Fitting Language {params[0]}")
            scores = fit_optimal_curve(params)
            pickle.dump(scores, open(f"results/learnability_scores/{params[0]}.p", "wb"))
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(fit_optimal_curve, items)
