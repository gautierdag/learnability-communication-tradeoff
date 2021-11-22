import pickle
import os
from typing import Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import expon, beta, multivariate_normal
from ck_blahut_arimoto import ck_blahut_arimoto_ib
from tqdm import tqdm

np.seterr(divide='ignore')


def fit_gaussians(data: pd.DataFrame, cov_reg: float = 1e-5) \
        -> Tuple[np.ndarray, List[Tuple[str, np.ndarray, np.ndarray]]]:
    """ Fit Gaussians to the given data conditioned on colour terms.

    Args:
        data: Data points of colour term and colour stimulus observations
        cov_reg: The covariance regularisation constant
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
        The distribution p(W, C|H)
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
        x_data, y_data = subset[vars_name[0]],  subset[vars_name[1]]
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


def sample_from_prior(pC, data, nk):
    if pC is None:
        return data

    return data.sample(nk, replace=True, weights=pC, axis=0, random_state=42, ignore_index=True)


color_data_integer = pd.read_csv("wcs/term.txt", sep="\t", names=["language", "speaker", "chip", "word"])
lang_id_lang = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1], index_col=0, names=["id", "language"])
chip_data = pd.read_csv("wcs/chip.txt", sep="\t", index_col=0, names=["row", "col", "pos"])

chip2lab = pd.read_csv("wcs/cnum-vhcm-lab-new.txt", sep="\t", header=0, index_col="#cnum")
chip2lab = chip2lab.sort_values(by="#cnum")[["L*", "a*", "b*"]]
color_data_integer = color_data_integer.merge(chip2lab, left_on="chip", right_on="#cnum", how="left")
covariance_prior = color_data_integer[["L*", "a*", "b*"]].cov().to_numpy()

C = 330  # Number of colour chips

inf_loss = []
complexity = []
progress = tqdm(color_data_integer.groupby('language'))

pC = None
if os.path.exists("pC.p"):
    pC = pickle.load(open("pC.p", "rb"))


# Array to store fitted model params
models = []

for i, (language, elicitations) in enumerate(progress):
    nd = elicitations.shape[0]  # Number of datapoints for this language
    nw = elicitations["word"].nunique()  # Number of unique colour terms
    lang_str = lang_id_lang.loc[language]["language"]  # human-readable name for the language
    k = 1 if pC is not None else 1  # The number of division to test
    for n in np.linspace(5, nd, k, dtype=int):
        if k == 1:
            sample = elicitations
        else:
            sample = sample_from_prior(pC, elicitations[:C], n)
        pWC_H, ms = learn_language(sample)

        # plot_gaussians(elicitations, ms)
        # plt.show()

        pC_H = pWC_H.sum(axis=0)
        pW_CH = pWC_H / pC_H

        pH = 1 / n

        ### COMPUTE p(c) IF DOESNT EXIST ###
        plC = None
        if pC is None:
            _, q_wct = ck_blahut_arimoto_ib(pWC_H, 1, pW_CH, "kl-divergence")
            plC = q_wct.sum(axis=0).sum(axis=1)
            plC /= plC.sum()

        models.append((
            (language, lang_str),
            (n, nd, nw),
            plC,
            pH,
            pWC_H,
            ms
        ))

### COMPUTE COLOUR PRIOR p(c) ###
if pC is None:
    pC = np.zeros((C, ))
    L = 0
    for model in models:
        _, language = model[0]
        if language in ["Amuzgo", "Camsa", "Candoshi", "Chayahuita", "Chiquitano", "Eastern Cree", "Carib",
                        "Ifugao", "Micmac", "Nahuatl", "Papago", "Slave", "Tacana", "Central Tarahumara",
                        "W. Tarahumara"]:
            continue
        plC = model[2]
        pC += plC
        L += 1
    pC /= L
    pickle.dump(pC, open("pC.p", "wb"))
    chip_data["row"] = chip_data["row"].apply(lambda x: ord(x) - 65)
    grid = np.zeros((8, 40))
    for i, row, col, pos in chip_data.itertuples():
        if col == 0: continue
        grid[row - 1, col - 1] = pC[i - 1]

    plt.imshow(np.flip(grid))
    plt.show()


for model in models:
    _, (n, _, _), plC, pH, pWC_H, _ = model

    ### COMPUTE INFORMATION LOSS (log likelihood) ###

    ll = np.log(pWC_H * pH)
    ll[ll == -np.inf] = 0.0
    inf_loss.append(-ll.sum() / n)

    ### COMPUTE COMPLEXITY (mutual information) ###
    pC_H = pWC_H.sum(axis=0)
    lratio = np.log(pC_H) - np.log(pC)
    lratio[lratio == -np.inf] = 0

    complexity.append(np.sum(pC_H * pH * lratio))

plt.plot(complexity, inf_loss, '.')
plt.xlabel("Complexity")
plt.ylabel("Information Loss")
plt.show()
