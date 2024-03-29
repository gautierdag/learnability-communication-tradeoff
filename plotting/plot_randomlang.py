import os

import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

seed = 42
plt.rc('font', size=18)
lids = [2, 32, 35, 108]

plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('figure', titlesize=24)  # fontsize of the figure title

path = f"output/som/{seed}/randomlang-randominit"
opath = f"output/som/{seed}/lang"
odict = pickle.load(open(os.path.join(opath, "scores_dict.p"), "rb"))
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
lang_strs = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1],
                        header=None, index_col=0, names=["id", "language"])

handles = []
labels = []
fig, ax = plt.subplots()
ax.set_xlabel(r"Complexity")
ax.set_ylabel("Reconstructive Error")
ax.set_title("Sampling Prior in English")
for i, lid in enumerate(lids):
    # dict_path = os.path.join(path, init_type)
    dict_path = path
    scores_dict = pickle.load(open(os.path.join(dict_path, "scores_dict.p"), "rb"))
    scores = scores_dict[lid]

    X, Y = scores[:-1, :2].T
    U, V = np.diff(scores[:, :2], axis=0).T
    ax.quiver(X, Y, U, V,
              angles="xy",
              scale_units="xy",
              scale=1,
              width=0.005,
              headwidth=2,
              color=colors[i % len(colors)],
              )
    ax.scatter(
        X, Y, s=6, edgecolor="white", linewidth=0.5, color=colors[i % len(colors)]
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
          width=0.005,
          headwidth=2,
          color="k",
          )
ax.scatter(
    X, Y, s=6, edgecolor="white", linewidth=0.5, color="k"
)
handles.append(Line2D([], [],
                      color="white",
                      markerfacecolor="k",
                      marker="o",
                      markersize=10))
labels.append("Original")

# plt.legend(handles, labels)
plt.title(f"{lang_strs.loc[lid, 'language']}")
fig.tight_layout()
fig.savefig(os.path.join(path, "learning_trajectories.pdf"))
plt.show()
