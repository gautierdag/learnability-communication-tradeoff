import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from learnability import GaussianLanguageModel, plot_color_prior

use_prior = False
seed = 42

adult_model = GaussianLanguageModel()
adult_model.learn_languages()

fig, ax = plt.subplots()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
lang_strs = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1],
                        header=None, index_col=0, names=["id", "language"])

ploted_lids = [2, 32, 35, 108]
language_ids = list(range(1, 111, 1))
sample_range = (
        list(range(1, 25, 1))
        + list(range(25, 50, 5))
        + list(range(50, 100, 10))
        + list(range(100, 220, 20))
        + list(range(250, 1000, 50))
        + list(range(1000, 2100, 100))
        + list(range(3000, 10001, 1000))
        + list(range(20000, 100001, 10000))
)
n_range = np.array(sample_range)

handles = []
labels = []

scores = {}
for i, lid in enumerate(language_ids):
    if use_prior:
        with open(f"output/learnability/{seed}/{lid}/scores_h.npy", "rb") as f:
            scores[lid] = np.load(f)
    else:
        with open(f"output/learnability/{seed}/{lid}/scores.npy", "rb") as f:
            scores[lid] = np.load(f)

    scores_array = scores[lid]

    ax.quiver(*scores_array[:-1].T, *np.diff(scores_array, axis=0).T,
              angles='xy', scale_units='xy', scale=1, alpha=len(adult_model.models_params[lid])/80,
              width=0.002, headwidth=1, headlength=0.01, color="grey"
              )
    # ax.scatter(scores_array[:, 0], scores_array[:, 1], s=4, c="grey",
    #            edgecolor="white", linewidth=0.25)

for i, lid in enumerate(ploted_lids):
    scores_array = np.array(scores[lid])
    ax.quiver(*scores_array[:-1].T, *np.diff(scores_array, axis=0).T,
              angles='xy', scale_units='xy', scale=1,
              width=0.01, headwidth=2, color=colors[i], edgecolor="w", linewidth=1
              )
    ax.scatter(scores_array[:, 0], scores_array[:, 1], s=6,
               edgecolor="white", linewidth=0.5)
    for j, n in enumerate(n_range):
        ax.text(*scores_array[j], n)
    handles.append(Line2D([], [], color="white", markerfacecolor=colors[i], marker="o", markersize=10))
    labels.append(lang_strs.loc[lid, "language"])

ax.legend(handles, labels)
ax.set_xlabel("Complexity; $I(W, C)$ bits")
ax.set_ylabel("Information Loss; $D[P_M || P_H]$ bits")
fig.tight_layout()
# fig.savefig("cplx_inf_loss.pdf")

fig, ax = plt.subplots()

inf_loss_thres = 5
inf_losses = []
for lid in language_ids:
    scores_array = np.array(scores[lid])
    nw = len(adult_model.models_params[lid])
    nidx = np.argwhere(scores_array[:, 1] < inf_loss_thres)[0][0]
    inf_loss = scores_array[nidx][1]
    inf_loss_passed = n_range[nidx]
    ax.scatter(nw, inf_loss_passed, s=8 * inf_loss ** 2, c="grey", edgecolor="white", linewidth=1)
    inf_losses.append((nw, inf_loss_passed))

inf_losses = np.array(inf_losses)

for i, lid in enumerate(ploted_lids):
    scores_array = np.array(scores[lid])
    nw = len(adult_model.models_params[lid])
    nidx = np.argwhere(scores_array[:, 1] < inf_loss_thres)[0][0]
    inf_loss = scores_array[nidx][1]
    inf_loss_passed = n_range[nidx]
    ax.scatter(nw, inf_loss_passed, s=8 * inf_loss ** 2, c=colors[i], edgecolor="white", linewidth=1)

a = np.vstack([np.ones_like(inf_losses[:, 0]), inf_losses[:, 0]]).T
b = inf_losses[:, 1]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
lr = LinearRegression()
lr.fit(a, b)
ys = lr.predict(a)
r = pearsonr(b, ys)
print("R", r)
xs = np.linspace(-5, 85, 110)
ys = lr.coef_[0] + lr.coef_[1] * xs
ax.plot(xs, ys, "--", c="k")

ax.legend(handles, labels)
ax.set_xlabel("$|W|$; Number of Colour Terms ")
ax.set_ylabel("N; Number of Samples")
fig.tight_layout()
# fig.savefig("w_samples.pdf")

plt.show()


pc = pickle.load(open("pc.p", "rb"))

for lid in language_ids[3:]:
    for n in n_range:
        if n < 100:
            continue
        with open(f"output/learnability/{seed}/{lid}/{n}.npy", "rb") as f:
            model = np.load(f)
        fig, ax = plt.subplots(3, 1)
        plot_color_prior(pc, ax[0])
        fig.suptitle(f"N = {n}")
        ax[0].set_ylabel("Colour Prior")
        plot_color_prior(adult_model.models[lid].to_numpy().sum(axis=0), ax[1])
        ax[1].set_ylabel("Adult Model")
        plot_color_prior(model.sum(axis=0), ax[2])
        ax[2].set_ylabel("Child Model")
        plt.show()

