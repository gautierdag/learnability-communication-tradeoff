import glob
import os

import imageio
import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.lines import Line2D

from noga.figures import mode_map
from som import SelfOrganisingMap

animate = True
lids = [2, 32, 35, 108]  # range(1, 111)
seed = 42
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

som = SelfOrganisingMap()
scores_dict = pickle.load(open(f"output/som/{seed}/scores_dict.p", "rb"))
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
lang_strs = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1],
                        header=None, index_col=0, names=["id", "language"])

# Plot frontiers against each other
fig = plt.figure()
for lid in lids:
    scores = pickle.load(open(f"frontier/learnability_languages/{lid}.p", "rb"))
    plt.plot(scores[0], scores[1], label=f"{lang_strs.loc[lid, 'language']} ({som.term_size[lid]})")
plt.legend()
plt.title("Optimal learning curves")
plt.xlabel("Complexity; $I(H, C)$ bits")
plt.ylabel("Information Loss; KL-Divergence bits")
fig.tight_layout()
fig.savefig(f"output/som/optimal_curves.pdf")
plt.show()

# Plot number of time steps vs information loss
fig, ax = plt.subplots()
for i, (lid, scores) in enumerate(scores_dict.items()):
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Information Loss; KL-Divergence bits")

    plt.plot(sample_range, scores[:, 1], c=colors[i % len(colors)],
             label=f"{lang_strs.loc[lid, 'language']} ({som.term_size[lid]})")
    plt.fill_between(sample_range, scores[:, 1] - scores[:, 3], scores[:, 1] + scores[:, 3],
                     color=colors[i % len(colors)], alpha=0.2)
plt.legend()
fig.tight_layout()
fig.savefig(f"output/som/{seed}/sample_inf_loss.pdf")
plt.show()

# Plot number of time steps vs complexity
fig, ax = plt.subplots()
for i, (lid, scores) in enumerate(scores_dict.items()):
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(r"Complexity; $I(W, C)$ bits")

    plt.plot(sample_range, scores[:, 0], c=colors[i % len(colors)],
             label=f"{lang_strs.loc[lid, 'language']} ({som.term_size[lid]})")
    plt.fill_between(sample_range, scores[:, 0] - scores[:, 2], scores[:, 0] + scores[:, 2],
                     color=colors[i % len(colors)], alpha=0.2)
plt.legend()
fig.tight_layout()
fig.savefig(f"output/som/{seed}/sample_complexity.pdf")
plt.show()

# Plot all selected LIDs on one plot without animation
handles = []
labels = []
fig, ax = plt.subplots()
for i, (lid, scores) in enumerate(scores_dict.items()):
    ax.set_xlabel(r"Complexity; $I(W, C)$ bits")
    ax.set_ylabel("Information Loss; KL-Divergence bits")
    ax.set_title(lang_strs.loc[lid, "language"])

    frontier = pickle.load(open(f"frontier/learnability_languages/{lid}.p", "rb"))
    ax.plot(frontier[0], frontier[1])

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
        X, Y, s=6, edgecolor="white", linewidth=0.5,
    )
    handles.append(Line2D([], [],
                          color="white",
                          markerfacecolor=colors[i % len(colors)],
                          marker="o",
                          markersize=10))
    labels.append(f"{lang_strs.loc[lid, 'language']} ({som.term_size[lid]})")
plt.legend(handles, labels)
fig.tight_layout()
fig.savefig(f"output/som/{seed}/learning_trajectories.pdf")
plt.show()

# Plot averaged learning trajectories
for i, (lid, scores) in enumerate(scores_dict.items()):
    fig, ax = plt.subplots()

    ax.set_xlabel("Complexity; $I(W, C)$ bits")
    ax.set_ylabel("Information Loss; KL-Divergence bits")
    ax.set_title(lang_strs.loc[lid, "language"])

    frontier = pickle.load(open(f"frontier/learnability_languages/{lid}.p", "rb"))
    ax.plot(frontier[0], frontier[1], color=colors[i % len(colors)])

    X, Y = scores[:-1, :2].T
    U, V = np.diff(scores[:, :2], axis=0).T

    if not animate:
        ax.quiver(X, Y, U, V,
                  angles="xy",
                  scale_units="xy",
                  scale=1,
                  width=0.005,
                  headwidth=2,
                  color=colors[i % len(colors)],
                  )
        ax.scatter(
            X, Y, s=6, edgecolor="white", linewidth=0.5
        )
        fig.tight_layout()
        fig.savefig(f"output/som/{seed}/{lid}/learning_traj_{lid}.pdf")
    else:
        artists = []
        for j, _ in enumerate(sample_range):
            q = ax.quiver(X[:j], Y[:j], U[:j], V[:j],
                          angles="xy",
                          scale_units="xy",
                          scale=1,
                          width=0.005,
                          headwidth=2,
                          color=colors[i % len(colors)],
                          animated=True)
            s = ax.scatter(
                X[:j], Y[:j], s=6, edgecolor="white", linewidth=0.5, color=colors[i % len(colors)]
            )
            artists.append([q, s])
        anim = ArtistAnimation(fig, artists, blit=True)
        fig.tight_layout()
        anim.save(f'output/som/{seed}/{lid}/learning_traj_{lid}.gif', dpi=300)

    plt.show()

# Plot mode maps
for i, (lid, scores) in enumerate(scores_dict.items()):
    if not os.path.exists(f"output/som/{seed}/{lid}/mode_maps/"):
        os.mkdir(f"output/som/{seed}/{lid}/mode_maps/")
    pt_s_arr = []
    for file in glob.glob(os.path.join(f"output/som/{seed}/{lid}/*.npy")):
        pt_s = np.load(file)
        n_samples = int(file.split(os.sep)[-1].split("_")[0])
        pt_s_arr.append((n_samples, pt_s))
    mode_maps = []
    for j, (n_samples, pt_s) in enumerate(sorted(pt_s_arr, key=lambda x: x[0])):
        mode_map(pt_s)
        plt.title(f"K={n_samples}")
        fig = plt.gcf()
        fig.tight_layout()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8
        )
        figure_size = tuple(
            np.array(fig.get_size_inches()[::-1] * fig.dpi, dtype=int)
        ) + (3,)
        image_from_plot = image_from_plot.reshape(figure_size)
        mode_maps.append(image_from_plot)
        fig.savefig(f"output/som/{seed}/{lid}/mode_maps/{j:03}.jpg")

    fp_in = f"output/som/{seed}/{lid}/mode_maps/*.jpg"
    fp_out = f"output/som/{seed}/{lid}/mode_map_{lid}.gif"
    images = list(map(lambda filename: imageio.imread(filename), sorted(glob.glob(fp_in))))
    imageio.mimsave(os.path.join(fp_out), images, duration=0.25)