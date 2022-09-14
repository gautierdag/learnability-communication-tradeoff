import os

import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

seed = 42
plt.rc('font', size=18)
lids = [2, 32, 35, 108]

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

path = f"output/som/{seed}/randominit"
opath = f"output/som/{seed}/lang"
# som = SelfOrganisingMap()
odict = pickle.load(open(os.path.join(opath, "scores_dict.p"), "rb"))
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
lang_strs = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1],
                        header=None, index_col=0, names=["id", "language"])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, lid in enumerate(lids):
    handles = []
    labels = []

    ax = axes[i]
    ax.set_xlabel(r"Complexity")
    if i == 0:
        ax.set_ylabel("Reconstructive Error")
    else:
        ax.set_ylabel("")
    for j, init_type in enumerate(["binary"]):
        dict_path = os.path.join(path, init_type)
        scores_dict = pickle.load(open(os.path.join(dict_path, "scores_dict.p"), "rb"))
        scores = scores_dict[lid]

        X, Y = scores[:-1, :2].T
        U, V = np.diff(scores[:, :2], axis=0).T
        ax.quiver(X, Y, U, V,
                  angles="xy",
                  scale_units="xy",
                  scale=1,
                  width=0.01,
                  headwidth=2,
                  color=colors[i % len(colors)],
                  )
        ax.scatter(
            X, Y, s=7, edgecolor="white", linewidth=0.5, color=colors[i % len(colors)]
        )
        ax.plot(X[0], Y[0], markersize=6, marker="x", color=colors[i % len(colors)])

        handles.append(Line2D([], [],
                              color="white",
                              markerfacecolor=colors[i % len(colors)],
                              marker="o",
                              markersize=10))
        labels.append("Random")

    scores = odict[lid]
    X, Y = scores[:-1, :2].T
    U, V = np.diff(scores[:, :2], axis=0).T
    ax.quiver(X, Y, U, V,
              angles="xy",
              scale_units="xy",
              scale=1,
              width=0.01,
              headwidth=2,
              color="grey",
              )
    ax.scatter(
        X, Y, s=7, edgecolor="white", linewidth=0.5, color="grey"
    )
    handles.append(Line2D([], [],
                          color="white",
                          markerfacecolor="grey",
                          marker="o",
                          markersize=10))
    labels.append("Standard")

    ax.set_title(f"{lang_strs.loc[lid, 'language']}")
    # ax.legend(handles, labels)
fig.tight_layout()
fig.savefig(os.path.join(path, f"randominit_{lid}.pdf"))
plt.show()
