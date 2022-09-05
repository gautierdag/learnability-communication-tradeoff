import glob
import os
import pickle


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from noga.figures import mode_map
from som import SelfOrganisingMap, sample_range

# betas = [32.0]
betas = [1.0942937012608183, 1.2226402776921537, 1.4439291955225915, 1.7171308728756145, 11.79415373832906]
seed = 42

path = f"output/som/{seed}/ce"
scores_dict = pickle.load(open(os.path.join(path, "scores_dict.p"), "rb"))
som = SelfOrganisingMap()
colors = matplotlib.cm.get_cmap('Dark2').colors
lang_strs = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1],
                        header=None, index_col=0, names=["id", "language"])
plt.rc('font', size=20)


# Plot evolution of colour naming distributions
# fig, axes = plt.subplots(2, 3, figsize=(12, 5))
# for i, s in enumerate([1, 200, 250, 350, 650, 50000]):
#     p_t_s = np.load(os.path.join(path, str(betas[3]), f"{s}_pt_s_all.npy")).mean(0)
#     p_t_s_som = som.pts[2]
#     ps = som.ps_universal
#
#     ax = axes[i // 3, i % 3]
#     mode_map(p_t_s, ps, ax=ax)
#     ax.set_title(rf"N={s}")
# plt.show()

# for i, beta in enumerate(betas):
#     p_t_s = np.load(os.path.join(path, str(beta), "50000_pt_s_all.npy")).mean(0)
#     file = glob.glob(f"frontier/q_matrices/{beta}*.npy")[0]
#     p_t_s_q = np.load(file)
#     ps = som.ps_universal
#
#     fig, axes = plt.subplots(2, 1)
#     ax = axes[0]
#     mode_map(p_t_s_q, ps, ax=ax)
#     if i == 0:
#         ax.set_ylabel("Data")
#
#     ax = axes[1]
#     mode_map(p_t_s, ps, ax=ax)
#     if i == 0:
#         ax.set_ylabel("SOM")
#
#     fig.tight_layout()
#     fig.savefig(os.path.join(path, f"mode_map_comparison_{beta}.pdf"))
#     # plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[1]
for i, beta in enumerate(betas):
    results = pickle.load(open("accuracies.pkl", "rb"))
    n_conv = pickle.load(open("n_conv.pkl", "rb"))
    plt.ylim((0, 1))
    means = np.array([np.mean(vals) for _, vals in sorted(results[f"ce_{beta}"].items(), key=lambda x: x[0])])
    stds = np.array([np.std(vals) for _, vals in sorted(results[f"ce_{beta}"].items(), key=lambda x: x[0])])
    ax.plot(sample_range, means, color=colors[i % len(colors)], label=f"{np.round(beta, 4)}")
    ax.fill_between(sample_range, means - stds, means + stds, color=colors[i % len(colors)], alpha=0.33)
ax.set_xlim([0, 20000])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Beta:")
ax.set_xlabel("Number of Samples")
ax.set_ylabel("Accuracy")

# Plot all selected LIDs on one plot without animation
handles = []
labels = []
ax = axes[0]
for i, beta in enumerate(betas):
    scores = scores_dict[beta]

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
        X, Y, s=6, edgecolor="white", linewidth=0.5, color=colors[i % len(colors)]
    )
    plt.plot()
    handles.append(Line2D([], [],
                          color="white",
                          markerfacecolor=colors[i % len(colors)],
                          marker="o",
                          markersize=10))
    labels.append(fr"${np.round(beta, 4)}$")
# plt.legend(handles, labels)
ax.set_xlabel(r"Complexity")
ax.set_ylabel("Reconstructive Error")
fig.tight_layout()
fig.savefig(os.path.join(path, "ce_plot.pdf"))
plt.show()
