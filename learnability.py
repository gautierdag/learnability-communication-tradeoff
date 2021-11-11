import pandas as pd
import numpy as np
import pickle
import os
from scipy.stats import expon, beta, multivariate_normal
from ck_blahut_arimoto import ck_blahut_arimoto_ib
import matplotlib.pyplot as plt
from tqdm import tqdm

np.seterr(divide='ignore')


color_data_integer = pd.read_csv("wcs/term.txt", sep="\t", names=["language", "speaker", "chip", "word"])
lang_id_lang = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1], index_col=0, names=["id", "language"])
chip_data = pd.read_csv("wcs/chip.txt", sep="\t", index_col=0, names=["row", "col", "pos"])

chip2lab = pd.read_csv("wcs/cnum-vhcm-lab-new.txt", sep="\t", header=0, index_col="#cnum")
chip2lab = chip2lab.sort_values(by="#cnum")[["L*", "a*", "b*"]]
color_data_integer = color_data_integer.merge(chip2lab, left_on="chip", right_on="#cnum", how="left")
covariance_prior = color_data_integer[["L*", "a*", "b*"]].cov().to_numpy()

cov_reg = 1e-5  # Regularisation parameter for Gaussian covariances
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
    N = elicitations.shape[0]
    lang_str = lang_id_lang.loc[language]["language"]
    W = max(pd.factorize(elicitations.word)[0]) + 1

    data = elicitations[["word", "L*", "a*", "b*"]]

    progress.set_postfix({"W": W, "Language": lang_str})

    ### COMPUTE p(c|w) ###
    pC_W = []
    for wt, subset in data.groupby("word"):
        subset = subset[["L*", "a*", "b*"]]
        mu, cov = subset.mean(), subset.cov()
        if subset.shape[0] == 1:
            cov = np.identity(3)
        cov += np.identity(3) * cov_reg

        proba = multivariate_normal(mu, cov).pdf(chip2lab)
        pC_W.append(proba / sum(proba))
    pC_W = np.array(pC_W)

    ### COMPUTE p(w, c|h) ###
    pW_H = (elicitations["word"].value_counts() / N).to_numpy().reshape((W, 1))
    pWC_H = pC_W * pW_H
    pC_H = pWC_H.sum(axis=0)
    pW_CH = pWC_H / pC_H

    pH = expon.cdf(N, 0.01)

    ### COMPUTE p(c) IF DOESNT EXIST ###
    plC = None
    if pC is None:
        _, q_wct = ck_blahut_arimoto_ib(pWC_H, 1, pW_CH, "entropy")
        plC = q_wct.sum(axis=0).sum(axis=1)

    models.append((
        (language, lang_str),
        N,
        plC,
        pC_W,
        pW_H,
        (mu, cov)
    ))

pickle.dump(models, open("models.p", "wb"))

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
    _, N, plC, pC_W, pW_H, _ = model
    pWC_H = pC_W * pW_H
    pC_H = pWC_H.sum(axis=0)

    ### COMPUTE INFORMATION LOSS (log likelihood) ###
    inf_loss.append(-1 / N * np.log(pWC_H * pH).sum())

    ### COMPUTE COMPLEXITY (mutual information) ###
    lratio = np.log(pC_H / pC)
    lratio[lratio == -np.inf] = 0

    complexity.append(np.sum(pC_H * pH * lratio))

plt.plot(complexity, inf_loss, '.')
plt.show()
