import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from convergence import evaluate_convergence_model
from som import SelfOrganisingMap

seed = 42
lscores = pickle.load(open(f"output/som/{seed}/scores_dict.p", "rb"))
sbetas, snum_words, sscores = pickle.load(open(f"output/som/{seed}/ce/suboptimal_scores_dict.p", "rb"))
betas, num_words, scores = pickle.load(open(f"output/som/{seed}/ce/optimal_scores_dict.p", "rb"))

lids = list(lscores)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
lang_strs = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1],
                        header=None, index_col=0, names=["id", "language"])
som = SelfOrganisingMap()

path = f"output/som/{seed}/ce/"
if not os.path.exists(path):
    os.mkdir(path)

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

# Plot suboptimal learning trajectories and their real-world counter-parts
for lid in lids:
    nid = [int(i) for i, ld in sbetas.items() if int(ld) == lid][0]
    s = sscores[nid]
    c = colors[nid % len(colors)]

    # Plot inf-loss vs number of samples
    fig, ax = plt.subplots()
    for c, label, s in [("red", "subopt", sscores[nid]), (c, "orig", lscores[lid])]:
        ax.plot(sample_range, s[:, 1], c=c, label=label)
        ax.fill_between(sample_range, s[:, 1] - s[:, 3], s[:, 1] + s[:, 3],
                        color=c, alpha=0.2)
    ax.legend()
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Information Loss; KL-Divergence bits")
    ax.set_title(f"Real-world vs sub-optimal\n{lang_strs.loc[lid, 'language']}")
    fig.tight_layout()
    fig.savefig(os.path.join(path, f"{lid}", "inf_loss_comparison.pdf"))
    plt.show()

    # Plot convergence vs number of samples
    fig, ax = plt.subplots()

    accs = []
    for n_sample in sample_range:
        p_t_s = np.load(os.path.join(path, f"{lid}", f"{n_sample}_pt_s.npy"))
        acc = evaluate_convergence_model(som, {lid: p_t_s}, [lid])[lid]
        accs.append(acc)
    ax.plot(sample_range, accs, c="red", label="subopt")

    accs = []
    data = som.term_data
    data = data[data["language"].isin([lid])]
    data = data[~pd.isna(data["word"])]
    chip_group = data.groupby("chip")
    for n_sample in sample_range:
        p_t_s = np.load(os.path.join(f"output/som/{seed}", f"{lid}", f"{n_sample}_pt_s.npy"))
        correct = 0
        ts = np.argmax(p_t_s, axis=1)
        for cid, chip in chip_group:
            if ts[cid - 1] == som.word_map[lid].inverse[chip['word'].mode()[0]]:
                correct += 1
        acc = correct / 330
        accs.append(acc)
    ax.plot(sample_range, accs, c=c, label="orig")

    ax.legend()
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Real-world vs sub-optimal\n{lang_strs.loc[lid, 'language']}")
    fig.tight_layout()
    fig.savefig(os.path.join(path, f"{lid}", "convergence_comparison.pdf"))
    plt.show()

# Plot number of time steps vs information loss
fig, ax = plt.subplots()
for i, s in scores.items():
    ax.plot(sample_range, s[:, 1], c=colors[i % len(colors)],
            label=rf"$\beta=${betas[i]:.4f}; $K=${num_words[i]}")
    ax.fill_between(sample_range, s[:, 1] - s[:, 3], s[:, 1] + s[:, 3],
                    color=colors[i % len(colors)], alpha=0.2)
ax.legend()
ax.set_title("CE-optimal language encoders")
ax.set_xlabel("Number of samples")
ax.set_ylabel("Information Loss; KL-Divergence bits")
fig.tight_layout()
fig.savefig(os.path.join(path, "samples_inf_loss.pdf"))
plt.show()

# Plot learning trajectories
handles = []
labels = []
prev_num_words = 0
for i, s in scores.items():
    if num_words[i] == prev_num_words:
        continue

    prev_num_words = num_words[i]

    plt.quiver(
        *s[:-1, :2].T,
        *np.diff(s[:, :2], axis=0).T,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        headwidth=2,
        color=colors[i % len(colors)]
    )
    plt.scatter(
        s[:, 0], s[:, 1], s=6, edgecolor="white", linewidth=0.5, color=colors[i % len(colors)]
    )
    handles.append(Line2D([], [],
                          color="white",
                          markerfacecolor=colors[i % len(colors)],
                          marker="o",
                          markersize=10))
    labels.append(rf"$\beta=${betas[i]:.4f}; $K=${num_words[i]}")
plt.xlabel("Complexity; $I(H, C)$ bits")
plt.ylabel("Information Loss; KL-Divergence bits")
plt.title("Learning trajectories for communicatively optimal systems")
plt.legend(handles, labels)
plt.gcf().tight_layout()
plt.savefig(os.path.join(path, "trajectories.pdf"))
plt.show()
