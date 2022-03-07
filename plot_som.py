import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
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

# Plot averaged learning trajectories
for i, (lid, scores) in enumerate(scores_dict.items()):
    fig, ax = plt.subplots()

    ax.set_xlabel("Complexity; $I(W, C)$ bits")
    ax.set_ylabel("Information Loss; KL-Divergence bits")
    ax.set_title(lang_strs.loc[lid, "language"])

    frontier = pickle.load(open(f"frontier/learnability_languages/{lid}.p", "rb"))
    ax.plot(frontier[0], frontier[1])

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
        fig.savefig(f"output/som/{seed}/{lid}/learning_traj.pdf")
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
        anim.save(f'output/som/{seed}/{lid}/learning_traj.gif')

    plt.show()