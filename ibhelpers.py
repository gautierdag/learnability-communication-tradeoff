import numpy as np
import ast
from ck_blahut_arimoto import ck_blahut_arimoto_ib, my_kl
from math import floor, log10

PRECISION = 1e-16

# score q using KL divergence for distortion
def score_q_kl(p_xy, q):
    result, q_xyt = ck_blahut_arimoto_ib(p_xy, 1, q, divergence=my_kl, max_iters=0)
    return result


# fit IB model


def fit_ib(p_xy, qinit, focalbeta, betas, tol=0.05, verbose=0):
    qs = [[] for b in betas]
    ibs = [[] for b in betas]
    qseq = []
    qseqresults = []
    qprev = qinit
    p_x = np.sum(p_xy, axis=1)

    qsm = mergecols(qinit, tol=tol)
    prevresultsize = qsm.shape[1]

    for i, beta in enumerate(betas):
        result, q_xyt = ck_blahut_arimoto_ib(p_xy, beta, qprev, max_iters=1000)
        q_xt = np.sum(q_xyt, axis=1)
        qprev = q_xt / q_xt.sum(axis=1, keepdims=True)
        qs[i] = qprev
        # score with my_kl
        result = score_q_kl(p_xy, qprev)

        qsm = mergecols(qprev, tol=tol)

        # @TODO: ADD SAVING HERE -> maybe including seed in filepath as well?
        # np.save(
        #     f"output/frontier_qs/b{beta}_r{result.rate}_d{result.distortion}_l{qsm.shape[1]}",
        #     qsm,
        # )

        # NB: compute minimum expected length using merged matrix
        e_len = ibsoln_elen(qsm, p_x)
        ibs[i] = (result.rate, result.distortion, e_len)

        resultsize = qsm.shape[1]
        if resultsize < prevresultsize:
            qseq.append(qprev)
            qseqresults.append((result.rate, result.distortion))
            prevresultsize = resultsize

        if verbose:
            print(str(beta) + ":" + str(resultsize))

    ind = np.argmin(abs(betas - focalbeta))
    q = qs[ind]
    return q, betas[ind], ibs, ibs[ind], qseq, qseqresults, qs


# --------------------------
# helper functions
# --------------------------


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
            qm[:, -1:] = qm[:, -1:] + q[:, i + 1 : i + 2]
        else:
            qm = np.c_[qm, q[:, i + 1 : i + 2]]

    return qm


def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        # put `first` in its own subset
        yield [[first]] + smaller


flatten = lambda l: [item for sublist in l for item in sublist]


def partition2q(s):
    n = len(flatten(s))
    q = np.zeros((n, n))
    for i, c in enumerate(s):
        q[c, i] = 1
    while i < n:
        q[c, i] = 1
        i = i + 1

    q = q / q.sum(axis=1, keepdims=True)

    return q


# perhaps maximum entropy?


def cats2q(cats, n):
    if cats is None:
        return (None, None)
    lens = np.concatenate((np.ones(len(cats)), np.zeros(n - len(cats))))
    # leftout category assumed unmarked -- ie length 0
    leftout = list(np.setdiff1d(list(range(n)), flatten(cats)))
    if leftout:
        cats.append(leftout)
    q = np.zeros((n, n))
    for i, c in enumerate(cats):
        q[c, i] = 1
    remainder = n - i
    if remainder > 1:
        q[c, i] = 1 / (remainder)
        lastlen = lens[i]
        while i < n:
            q[c, i] = 1 / remainder
            lens[i] = lastlen
            i = i + 1

    q = q / q.sum(axis=1, keepdims=True)

    return q, lens


def zmlabel(categories, lens, items):
    subs = dict(
        {
            "(f)r": "r(f)",
            "abcrxyz": "srf",
            "(xyz)abcr": "sr(f)",
            "(abc)(rxyz)": "(s)(rf)",
            "(abc)rxyz": "(s)rf",
            "(abc)(r)(xyz)": "(s)(r)(f)",
            "(abc)(xyz)r": "(s)r(f)",
            "(xyz)(c)(b)(a)r": "(a)(b)(c)r(f)",
            "(r)(xyz)(c)(b)(a)": "(a)(b)(c)(r)(f)",
            "(xyz)(abcr)": "(sr)(f)",
        }
    )

    strfinal = ""
    strp = ["".join(map(str, items[c])) for c in categories]
    if lens[-1] == 0:
        strfinal = strp[-1]
        strp = strp[0:-1]
    if len(strp) > 0:
        lab = "(" + ")(".join(strp) + ")" + strfinal
    else:
        lab = strfinal
    if lab in subs:
        lab = subs[lab]

    return lab


def zlabel(categories, items):
    subs = dict(
        {
            "(r)(sf)": "(sf)(r)",
            "(abcrxyz)": "(srf)",
            "(xyz)(abcr)": "(sr)(f)",
            "(abc)(rxyz)": "(s)(rf)",
            "(abc)(r)(xyz)": "(s)(r)(f)",
            "(abcr)(xyz)": "(sr)(f)",
            "(a)(b)(c)(rxyz)": "(a)(b)(c)(rf)",
        }
    )
    categories.sort()
    strp = ["".join(map(str, items[c])) for c in categories]
    lab = "(" + ")(".join(strp) + ")"
    if lab in subs:
        lab = subs[lab]

    return lab


# XXX: should be able to remove and use zlabel instead
def partition_label(p):
    print(p)
    subs = dict({"(r)(sf)": "(sf)(r)"})
    strp = ["".join(map(str, c)) for c in p]
    lab = "(" + ")(".join(strp) + ")"
    if lab in subs:
        lab = subs[lab]
    return lab


def partition_label_alternative(p):
    strp = ["".join(map(str, c)) for c in p]
    lab = "[" + "],[".join(strp) + "]"
    return lab


def from_np_array(array_string):
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def dotsizes(cnt):
    dotmax = 138
    dotmin = 6
    fmax = 140
    interval = (dotmax - dotmin) / (fmax - 1)
    dss = np.arange(dotmin, dotmax, interval)
    return dss[cnt]


# compute expected length assuming zero-coding of one category
# attested systems c may be incomplete -- left out category assumed to be zero-coded
# enumerated systems will be complete -- need to specify index of unmarked category


def exp_len(q, p_x, lens):
    nq = np.copy(q)
    nq[:, lens == 0] = 0
    e_len = np.sum(p_x * np.sum(nq, axis=1))
    return e_len


# use -1 to indicate no zero-coding


def make_lens(i, nc, n):
    l = np.ones(n)
    if i >= 0:
        if i < nc - 1:
            l[i] = 0
        else:
            l[i:n] = 0
    return l


def ibsoln_elen(qsm, p_x):
    nc = qsm.shape[1]
    qsm = qsm / qsm.sum(axis=0, keepdims=True)
    els = np.zeros(nc)
    for i in np.arange(nc):
        l = np.ones(nc)
        l[i] = 0
        els[i] = exp_len(qsm, p_x, l)

    return min(els)


def smarter_round(sig):
    def rounder(x):
        if x < 0.000000001:
            return 0

        offset = sig - floor(log10(abs(x)))
        initial_result = round(x, offset)
        if str(initial_result)[-1] == "5" and initial_result == x:
            return round(x, offset - 2)
        else:
            return round(x, offset - 1)

    return rounder


def naturalness(channels, meanings):
    # \sum_w argmin_s KL[P(m|w) || P(m|s)]
    n = []
    for w in mergecols(channels).transpose():
        w = w / sum(w)
        # print(w)
        naturalness = []
        for m in meanings:
            # print(np.round(m, 2))
            naturalness.append(my_kl(w, m))
        n.append(min(naturalness))
    return n


def xlogx(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > PRECISION, v * np.log2(v), 0)


def H(p, axis=None):
    """Entropy"""
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    """mutual information, I(X;Y)"""
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)


def DKL(p, q, axis=None):
    """KL divergences, D[p||q]"""
    return (xlogx(p) - np.where(p > PRECISION, p * np.log2(q + PRECISION), 0)).sum(
        axis=axis
    )


def gNID(q1, qstar, pX):
    # Lifted from Noga https://github.com/nogazs/ib-color-naming/blob/843af712df35bf13333a5175971ddd710b68cfac/tools.py
    pW_X = mergecols(q1).T / mergecols(q1).sum(axis=1)
    pV_X = mergecols(qstar).T / mergecols(qstar).sum(axis=1)
    pXW = pW_X * pX
    pWV = pXW.dot(pV_X.T)
    pWW = pXW.dot(pW_X.T)
    pVV = (pV_X * pX).dot(pV_X.T)
    return 1 - MI(pWV) / (np.max([MI(pWW), MI(pVV)]))
