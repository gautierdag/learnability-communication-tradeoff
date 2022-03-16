import multiprocessing
import os.path
import pickle
import sys
import numpy as np

import matplotlib.pyplot as plt
from som import SelfOrganisingMap


def mergecols(q, tol=0.05):
    """
    merge cols that represent near-identical words
    """
    zerotol = 1e-20
    # convert from p(x,w) to p(x|w)
    colsums = np.sum(q, axis=0, keepdims=True)
    q = q[:, np.where(colsums > zerotol)[1]]
    q = q / np.sum(q)
    qn = q / np.sum(q, axis=0, keepdims=True)
    colorder = qn[0, :].argsort()
    # reorder so columns to be merged are adjacent
    q = q[:, colorder]
    qn = qn[:, colorder]
    qn_diff = np.diff(qn, n=1, axis=1)
    qn_diffs = np.max(abs(qn_diff), axis=0)

    qm = q[:, 0:1]
    for i in np.arange(len(qn_diffs)):
        if qn_diffs[i] < tol:
            qm[:, -1:] = qm[:, -1:] + q[:, i + 1:i + 2]
        else:
            qm = np.c_[qm, q[:, i + 1:i + 2]]

    return qm


def stochastic_bottleneck(p_x, p_y_x, q_t_x, beta, maxiters=10000, verbose=False):
    """
    The stochastic information bottleneck.
        The distortion metric is the cross-entropy
    INPUT:
    OUTPUT:
    """
    iters = 0
    d0 = 0

    p_xy = (p_y_x * p_x[:, np.newaxis]) / (p_y_x * p_x[:, np.newaxis]).sum()
    q_t = np.matmul(p_x, q_t_x)
    q_y_t = np.matmul(q_t_x.T, p_xy) / q_t[:, np.newaxis]
    d = -1 * np.matmul(p_y_x, np.log2(q_y_t.T))

    while not np.isclose(np.matmul(p_x, q_t_x * d).sum(), d0) and iters < maxiters:
        iters += 1
        d0 = np.matmul(p_x, q_t_x * d).sum()
        q_xt = q_t * np.exp2(-1 * beta * d)
        q_xt = q_xt / q_xt.sum()
        q_t_x = q_xt / q_xt.sum(axis=1, keepdims=True)
        q_t = np.matmul(p_x, q_t_x)
        q_y_t = np.matmul(q_t_x.T, p_xy) / q_t[:, np.newaxis]
        d = -1 * np.matmul(p_y_x, np.log2(q_y_t.T))
        if verbose:
            print(iters, (p_xy * d).sum(), d0)

    q_xt = q_t_x * p_x[:, np.newaxis]
    rate = (q_xt * np.log2(q_xt / np.outer(p_x, q_t))).sum()
    distortion = np.matmul(p_x, q_t_x * (
                d + (np.matmul(p_y_x, np.log2(p_y_x.T)) * np.eye(p_x.shape[0])).sum(axis=1, keepdims=True))).sum()

    return q_t_x, rate, distortion


def rev_deterministic_annealing_IB(p_x, p_y_x, schedule, init, maxiters=10000, verbose=False):
    rates = []
    distortions = []
    qs = []

    # First pass
    q, r, d = stochastic_bottleneck(p_x, p_y_x, init, schedule[0], maxiters=maxiters)
    rates.append(r)
    distortions.append(d)
    qs.append(q)

    if verbose:
        print(schedule[0], np.round(r, 4), np.round(d, 4), mergecols(q).shape[1])

    for b in schedule[1:]:
        q, r, d = stochastic_bottleneck(p_x, p_y_x, q, b, maxiters=maxiters)
        rates.append(r)
        distortions.append(d)
        qs.append(mergecols(q))
        if verbose:
            print(b, np.round(r, 4), np.round(d, 4), mergecols(q).shape[1])

    return qs, (rates, distortions)


def fit_optimal_curve(language_id: int, ps: np.ndarray, a: float, path: str = None):
    q = (1 - a) * np.eye(ps.shape[0])
    q += (1 - np.eye(ps.shape[0])) * a / (ps.shape[0] - 1)

    q0 = np.eye(ps.shape[0])
    betas = np.array([2 ** x for x in np.arange(3, 0, -0.01)])
    qss, scores = rev_deterministic_annealing_IB(ps, q, betas, q0, verbose=True)

    if path is not None:
        pickle.dump(scores, open(os.path.join(path, f"{language_id}.p"), "wb"))
    return scores


def func(args):
    language_id = args[0]
    ps = args[1]
    a = args[2]
    path = args[3]

    q = (1 - a) * np.eye(ps.shape[0])
    q += (1 - np.eye(ps.shape[0])) * a / (ps.shape[0] - 1)

    q0 = np.eye(ps.shape[0])
    betas = np.array([2 ** x for x in np.arange(3, 0, -0.01)])
    qss, scores = rev_deterministic_annealing_IB(ps, q, betas, q0, verbose=True)

    if path is not None:
        pickle.dump(scores, open(os.path.join(path, f"{language_id}.p"), "wb"))
    return scores


if __name__ == '__main__':
    alpha = 0.1  # Uniform noise level
    beta = 0.1  # Weight of the capacity achieving prior for ps
    workers = 4
    save_path = "frontier/learnability_languages/"
    lids = [2, 32, 35, 108]

    if not os.path.exists("frontier"):
        os.mkdir("frontier")
    if not os.path.exists("frontier/learnability_languages"):
        os.mkdir("frontier/learnability_languages")

    # Create a SOM with arbitrary params to load data and calculate data distributions
    som = SelfOrganisingMap()

    if workers is not None:
        params = [(lid, np.squeeze((1 - beta) * ps + beta * som.ps_universal), alpha, save_path)
                  for lid, ps in som.ps.items() if lid in lids]
        with multiprocessing.Pool(processes=workers) as p:
            p.map(func, params)
    else:
        for lid, pt_s in som.pt_s.items():
            if len(sys.argv) > 1 and lid != int(sys.argv[1]):
                continue
            if lid not in [2, 32, 35, 108]: continue
            print(f"Fitting Language {lid}")
            # ps_l = (pt_s * som.ps_universal).sum(1)
            ps_l = (1 - beta) * som.ps[lid] + beta * som.ps_universal
            fit_optimal_curve(lid, ps_l, alpha, save_path)
