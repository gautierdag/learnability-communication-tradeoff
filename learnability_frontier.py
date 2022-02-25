import pickle

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from ibhelpers import fit_ib
from som import SelfOrganisingMap


def fit_optimal_curve(q):
    q = q.T
    num_cols = q.shape[1]
    if num_cols < 330:
        duplicate_vector = np.ones(num_cols, dtype=int)
        duplicate_vector[-1] = 331 - num_cols
        q = np.repeat(q, duplicate_vector, axis=1)
    q[:, 330 - duplicate_vector[-1]:] /= duplicate_vector[-1]
    q[np.isnan(q)] = 0
    # uniform dist for q va
    q[q.sum(1) == 0] = 1 / q.shape[0]

    q0 = np.eye(pts.shape[1])
    focalbeta = 1
    betas = np.array([2 ** x for x in np.arange(5, -2, -0.5)])
    return fit_ib(q, q0, focalbeta, betas, verbose=1, divergence="entropy")


if __name__ == '__main__':
    a = 0.5  # Portion of the probability mass that should be distributed uniformly across non-diagonals in p(c_hat|c)
    multiproc = False
    workers = 32

    # Create a SOM with arbitrary params to load data and calculate data distributions
    som = SelfOrganisingMap(1, 0.5, 1, 1)

    if not multiproc:
        for lid, pts in som.pts.items():
            q, beta, ibscores, qresult, qseq, qseqresults, allqs = fit_optimal_curve()

            pickle.dump(ibscores, open(f"scores_{lid}.p", "wb"))
    else:
        with multiprocessing.Pool() as pool:
            pass