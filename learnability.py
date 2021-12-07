import pickle
import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import expon, beta, multivariate_normal
from ck_blahut_arimoto import ck_blahut_arimoto_ib
from tqdm import tqdm


def fit_gaussians(data: pd.DataFrame, cov_reg: float = 1e-5) \
        -> Tuple[np.ndarray, List[Tuple[str, np.ndarray, np.ndarray]]]:
    """ Fit Gaussians to the given data conditioned on colour terms.

    Args:
        data: Data points of colour term and colour stimulus observations
        cov_reg: The covariance regularisation constant

    Returns:
        Tuple of the conditional distribution p(C|W) and the gaussian model params for each w in W
    """
    # By fitting conditional Gaussians compute the distribution p(C|W)
    pc_w = []
    model_params = []
    for ct, subset in data.groupby("word"):
        subset = subset[["L*", "a*", "b*"]]

        mu, cov = subset.mean().to_numpy(), subset.cov().to_numpy()
        if subset.shape[0] == 1:
            cov = np.identity(3)
        cov += np.identity(3) * cov_reg
        proba = multivariate_normal(mu, cov).pdf(chip2lab)

        model_params.append((ct, mu, cov))
        pc_w.append(proba / sum(proba))
    return np.array(pc_w), model_params


def learn_language(data: pd.DataFrame) -> np.ndarray:
    """ Learn the joint distribution of colour stimuli and terms given data.

    Args:
        data: Data containing all observations of colour terms and stimuli for the language

    Returns:
        The distribution p(W, C)
    """
    # Compute p(C|W)
    pc_w, models = fit_gaussians(data)

    # Compute p(W, C|H)
    pw_h = 1  # This assumes that meanings to colour terms are one-to-one
    pwc_h = pc_w * pw_h
    pwc_h /= pwc_h.sum()

    return pwc_h, models


def plot_gaussians(data: pd.DataFrame, models: List[Tuple[str, np.ndarray, np.ndarray]],
                   vars: Tuple[str, str] = (0, 2), axis: plt.Axes = None):
    """ Contour-plot of Gaussian distributions for each colour term and the corresponding data points.

    Args:
        data: Same dataframe that was used to fit each model in models
        models: Mean and covariance matrices for each Gaussian model fitted
        vars: Indices of which two variables from L*A*B* to select for plotting
        axis: Optional axis to draw on
    """
    vars = list(vars)
    vars_name = np.array(["L*", "a*", "b*"])[vars]

    if axis is None:
        fig, axis = plt.subplots()

    for i, (ct, subset) in enumerate(data.groupby("word")):
        subset = subset[vars_name]
        x_data, y_data = subset[vars_name[0]], subset[vars_name[1]]
        x_min, x_max = x_data.min() - 1, x_data.max() + 1
        y_min, y_max = y_data.min() - 1, y_data.max() + 1

        _, mu, cov = models[i]

        mu = mu[vars]
        cov = cov[vars, :]
        cov = cov[:, vars]

        n = 100
        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)
        x, y = np.meshgrid(x, y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        z = multivariate_normal(mu, cov).pdf(pos)
        axis.contour(x, y, z)
        axis.scatter(x_data.values, y_data.values, marker="o")
    axis.set_title("Fitted Gaussian contours")
    axis.set_xlabel(vars_name[0])
    axis.set_ylabel(vars_name[1])
    return axis


@dataclass
class Model:
    lang_id: int
    lang_str: str
    n_obs: int
    n_words: int
    pwc_h: np.ndarray
    plc: Optional[np.ndarray]


if __name__ == '__main__':
    np.seterr(divide='ignore')
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_min", type=int, help="The minimum number of samples to draw for each language.",
                        default=1)
    parser.add_argument("--n_step", type=int, help="The step rate of the number of samples to draw.",
                        default=1)
    parser.add_argument("--n_max", type=int, help="The maximum number of samples to draw for each language.",
                        default=100)
    parser.add_argument("-w", "--wcs_path", type=str, help="Path to the directory containing World Color Survey data.",
                        default="wcs")
    parser.add_argument("-l", "--language", type=str, help="Specify name of language to only generate samples for.")
    parser.add_argument("-o", "--output", type=str, help="Output path", default="")
    args = parser.parse_args()

    print("Running learnability problem")

    print(f"Loading WCS survey data from \"{args.wcs_path}\" - ", end="")
    wcs_path = args.wcs_path
    color_data_integer = pd.read_csv(os.path.join(wcs_path, "term.txt"),
                                     sep="\t", names=["language", "speaker", "chip", "word"])
    lang_id_lang = pd.read_csv(os.path.join(wcs_path, "lang.txt"),
                               sep="\t", usecols=[0, 1], index_col=0, names=["id", "language"])
    chip_data = pd.read_csv(os.path.join(wcs_path, "chip.txt"),
                            sep="\t", index_col=0, names=["row", "col", "pos"])
    chip2lab = pd.read_csv(os.path.join(wcs_path, "cnum-vhcm-lab-new.txt"),
                           sep="\t", header=0, index_col="#cnum")

    chip2lab = chip2lab.sort_values(by="#cnum")[["L*", "a*", "b*"]]
    color_data_integer = color_data_integer.merge(chip2lab, left_on="chip", right_on="#cnum", how="left")
    covariance_prior = color_data_integer[["L*", "a*", "b*"]].cov().to_numpy()
    print("done")

    C = chip2lab.shape[0]  # Number of colour chips
    N_min = args.n_min  # Number of samples to draw
    N_step = args.n_step  # Number of samples to draw
    N_max = args.n_max  # Number of samples to draw

    print("Trying to load colour prior p(C) - ", end="")
    pC = None
    if os.path.exists("pC.p"):
        pC = pickle.load(open("pC.p", "rb"))
        print("done")
    else:
        print("failed")
        print("Prior will be calculated during runtime.")

    print("Calculating p(W, C) for languages")
    models = []  # Array to store fitted model params
    progress = tqdm(color_data_integer.groupby('language'))
    for i, (lang_id, elicitations) in enumerate(progress):
        nd = elicitations.shape[0]  # Number of datapoints for this language
        nw = elicitations["word"].nunique()  # Number of unique colour terms
        lang_str = lang_id_lang.loc[lang_id]["language"]  # human-readable name for the language

        progress.set_postfix({"lang": lang_str, "nd": nd, "nw": nw})

        # Calculate joint distribution p(W,C|H)
        pWC_H, _ = learn_language(elicitations)

        # Compute prior for language p_l(C) if it doesn't exist
        plC = None
        if pC is None:
            pC_H = pWC_H.sum(axis=0)
            pW_CH = pWC_H / pC_H

            _, q_wct = ck_blahut_arimoto_ib(pWC_H, 1, pW_CH, "kl-divergence")

            plC = q_wct.sum(axis=0).sum(axis=1)
            plC /= plC.sum()

        # Sample N points from the joint distribution
        samples = []
        flat_joint = pWC_H.flatten()
        for N in np.arange(N_min, N_max, N_step):
            sample_idx = np.random.choice(np.arange(nw * C), (N,), True, flat_joint)
            samples.append(sample_idx)

        model = Model(lang_id, lang_str, nd, nw, pWC_H, plC)
        models.append((model, samples))

        # FOR TESTING; TODO: REMOVE LATER
        if i == 0: break

    # Compute colour prior p(C) if it doesn't exist
    if pC is None:
        print("Calculating prior - ", end="")
        pC = np.zeros((C,))
        L = 0
        for model, _ in models:
            if model.lang_id in ["Amuzgo", "Camsa", "Candoshi", "Chayahuita", "Chiquitano", "Eastern Cree", "Carib",
                                 "Ifugao", "Micmac", "Nahuatl", "Papago", "Slave", "Tacana", "Central Tarahumara",
                                 "W. Tarahumara"]:
                continue
            pC += model.plc
            L += 1
        pC /= L
        pickle.dump(pC, open("pC.p", "wb"))
        print("done")

        chip_data["row"] = chip_data["row"].apply(lambda x: ord(x) - 65)
        grid = np.zeros((8, 40))
        for i, row, col, pos in chip_data.itertuples():
            if col == 0: continue
            grid[row - 1, col - 1] = pC[i - 1]
        plt.imshow(np.flip(grid))
        plt.show()

    for model, samples in models:
        inf_loss = []
        complexity = []
        pWC_H = model.pwc_h
        pWC_H_flat = pWC_H.flatten()
        for sample in samples:
            # Simplicity prior
            terms = np.sort(np.unique(sample % model.n_words))  # Colour terms appearing in the sample
            n_unique = len(terms)
            pH = expon.pdf(len(sample), scale=pWC_H.shape[0])

            # Compute information loss (log likelihood)
            ll = np.log(pWC_H_flat[sample]) + np.log(pH)
            ll[ll == -np.inf] = 0.0
            inf_loss.append(ll.sum())

            # Compute complexity (mutual information)
            pC_H = pWC_H[terms, :].sum(axis=0)
            pC_H /= pC_H.sum()
            lratio = np.log(pC_H) - np.log(pC)
            lratio[lratio == -np.inf] = 0
            complexity.append(np.sum(pC_H * pH * lratio))

        plt.plot(complexity, inf_loss, '.')
    plt.xlabel("Mutual information")
    plt.ylabel("Log likelihood")
    plt.show()
