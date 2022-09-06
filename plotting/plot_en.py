import glob
import os

import imageio
import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.lines import Line2D
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from noga.figures import mode_map, WCS_CHIPS
from noga.tools import lab2rgb
from som import SelfOrganisingMap, sample_range

animate = False
animate_traj = False
seed = 42
plt.rc('font', size=18)

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

path = f"output/som/{seed}/en"
# som = SelfOrganisingMap()
corpus_dict = pickle.load(open(os.path.join(path, "corpus_dict.p"), "rb"))
elli_dict = pickle.load(open(os.path.join(path, "elli_dict.p"), "rb"))
uniform_dict = pickle.load(open(os.path.join(path, "uniform_dict.p"), "rb"))
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

handles = []
labels = []
fig, ax = plt.subplots()
ax.set_xlabel(r"Complexity")
ax.set_ylabel("Reconstructive Error")
ax.set_title("Sampling Prior in English")
for i, (sample_type, scores) in enumerate(zip(["Corpus", "Elicitation", "Uniform"],
                                              [corpus_dict[6], elli_dict[6], uniform_dict[6]])):
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
    plt.plot()
    handles.append(Line2D([], [],
                          color="white",
                          markerfacecolor=colors[i % len(colors)],
                          marker="o",
                          markersize=10))
    labels.append(sample_type)
plt.legend(handles, labels)
fig.tight_layout()
fig.savefig(os.path.join(path, "learning_trajectories.pdf"))
plt.show()
