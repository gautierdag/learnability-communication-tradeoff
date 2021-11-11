"""
The Blahut-Arimoto algorithm for solving the rate-distortion problem.
"""
from dit.rate_distortion.blahut_arimoto import _blahut_arimoto

import numpy as np


# use cross entropy for BA algorithm because the implementation inf blahut_arimoto.py goes wrong if we use KLD: think
# this is because of line 82  --  prev_d = 0 -- which causes troubles

# needed to define this because the dit version uses Distributions rather than numpy nd_arrays


def my_relative_entropy(dist1, dist2):
    """
    The cross entropy between `dist1` and `dist2`.

    Returns
    -------
    xh : float
        The cross entropy between `dist1` and `dist2`.

    """

    p1s = dist1
    q1s = dist2

    xh = -np.nansum(p1s * np.log2(q1s))
    return xh


def my_kl(dist1, dist2):
    """
    The KL divergence between `dist1` and `dist2`.

    Returns
    -------
    kld : float
        The KL divergence between `dist1` and `dist2`.

    """

    p1s = dist1
    q1s = dist2

    xh = -np.nansum(p1s * np.log2(q1s))
    e = -np.nansum(p1s * np.log2(p1s))
    kld = xh - e

    return kld


###############################################################################
# Information Bottleneck

# Adjusted the dit version to allow initialization


def ck_blahut_arimoto_ib(
    p_xy, beta, qinit, divergence=my_relative_entropy, max_iters=100
):  # pragma: no cover
    """
    Perform a robust form of the Blahut-Arimoto algorithms.

    Parameters
    ----------
    p_xy : np.ndarray
        The pmf to work with.
    beta : float
        The beta value for the optimization.
    q_y_x : np.ndarray
        The initial condition to work with.
    divergence : func
        The divergence measure to construct a distortion from: D(p(Y|x)||q(Y|t)).
    max_iters : int
        The maximum number of iterations.

    Returns
    -------
    result : RateDistortionResult
        The rate, distortion pair.
    q_xyt : np.ndarray
        The distribution p(x, y, t) which achieves the optimal rate, distortion.
    """
    p_x = p_xy.sum(axis=1)
    p_y_x = p_xy / p_xy.sum(axis=1, keepdims=True)

    def next_q_y_t(q_t_x):
        """
        q(y|t) = (\sum_x p(x, y) * q(t|x)) / q(t)
        """
        q_xyt = q_t_x[:, np.newaxis, :] * p_xy[:, :, np.newaxis]
        q_ty = q_xyt.sum(axis=0).T
        q_y_t = q_ty / q_ty.sum(axis=1, keepdims=True)
        q_y_t[np.isnan(q_y_t)] = 1
        return q_y_t

    def distortion(p_x, q_t_x):
        """
        d(x, t) = D[ p(Y|x) || q(Y|t) ]
        """
        q_y_t = next_q_y_t(q_t_x)
        if divergence == "entropy":
            lq_y_t = np.log2(q_y_t)
            distortions = np.array([-np.nansum(row * lq_y_t, axis=1) for row in p_y_x])
        else:
            distortions = np.asarray(
                [divergence(a, b) for a in p_y_x for b in q_y_t]
            ).reshape(q_y_t.shape)
        return distortions

    rd, q_xt = _blahut_arimoto(
        p_x=p_x,
        beta=beta,
        q_y_x=qinit,
        distortion=distortion,
        max_iters=max_iters,
    )

    q_t_x = q_xt / q_xt.sum(axis=1, keepdims=True)
    q_xyt = p_xy[:, :, np.newaxis] * q_t_x[:, np.newaxis, :]

    return rd, q_xyt
