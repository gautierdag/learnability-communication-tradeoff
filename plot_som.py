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
lids = [2, 32, 35, 108]  # range(1, 111)
seed = 42
plt.rc('font', size=16)


path = f"output/som/{seed}/lang"
spath = f"output/som/{seed}/suboptimal"
som = SelfOrganisingMap()
scores_dict = pickle.load(open(os.path.join(path, "scores_dict.p"), "rb"))
sscores_dict = pickle.load(open(os.path.join(spath, "scores_dict.p"), "rb"))
rotation = pickle.load(open("worst_qs_rotated.p", "rb"))
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
lang_strs = pd.read_csv("wcs/lang.txt", sep="\t", usecols=[0, 1],
                        header=None, index_col=0, names=["id", "language"])

# # Plot frontiers against each other
# fig = plt.figure()
# for lid in lids:
#     scores = pickle.load(open(f"frontier/learnability_languages/{lid}.p", "rb"))
#     plt.plot(scores[0], scores[1], label=f"{lang_strs.loc[lid, 'language']} ({som.term_size[lid]})")
# plt.legend()
# plt.title("Optimal learning curves")
# plt.xlabel("Complexity; $I(H, C)$ bits")
# plt.ylabel("Information Loss; KL-Divergence bits")
# fig.tight_layout()
# fig.savefig(os.path.join(path, "optimal_curves.pdf"))
# plt.show()
#
# # Plot number of time steps vs information loss
# fig, ax = plt.subplots()
# for i, (lid, scores) in enumerate(scores_dict.items()):
#     if lid not in lids: continue
#     ax.set_xlabel("Number of samples")
#     ax.set_ylabel("Information Loss; KL-Divergence bits")
#
#     plt.plot(sample_range, scores[:, 1], c=colors[i % len(colors)],
#              label=f"{lang_strs.loc[lid, 'language']} ({som.term_size[lid]})")
#     plt.fill_between(sample_range, scores[:, 1] - scores[:, 3], scores[:, 1] + scores[:, 3],
#                      color=colors[i % len(colors)], alpha=0.2)
# plt.legend()
# fig.tight_layout()
# fig.savefig(os.path.join(path, "sample_inf_loss.pdf"))
# plt.show()
#
# # Plot number of time steps vs complexity
# fig, ax = plt.subplots()
# for i, (lid, scores) in enumerate(scores_dict.items()):
#     if lid not in lids: continue
#
#     ax.set_xlabel("Number of samples")
#     ax.set_ylabel(r"Complexity; $I(W, C)$ bits")
#
#     plt.plot(sample_range, scores[:, 0], c=colors[i % len(colors)],
#              label=f"{lang_strs.loc[lid, 'language']} ({som.term_size[lid]})")
#     plt.fill_between(sample_range, scores[:, 0] - scores[:, 2], scores[:, 0] + scores[:, 2],
#                      color=colors[i % len(colors)], alpha=0.2)
# plt.legend()
# fig.tight_layout()
# fig.savefig(os.path.join(path, "sample_complexity.pdf"))
# plt.show()
#
# # Plot evolution of colour naming distributions
# fig, axes = plt.subplots(2, 3, figsize=(12, 5))
# for i, s in enumerate([1, 200, 250, 350, 650, 50000]):
#     p_t_s = np.load(os.path.join(path, str(2), f"{s}_pt_s_all.npy")).mean(0)
#     p_t_s_som = som.pts[2]
#     ps = 0.1 * som.ps_universal + 0.9 * som.ps[2]
#
#     ax = axes[i // 3, i % 3]
#     mode_map(p_t_s, ps, ax=ax)
#     ax.set_title(rf"N={s}")
#
# fig.tight_layout()
# fig.savefig(os.path.join(path, f"mode_map_evolution_{2}.pdf"))
# plt.show()
#
# # Plot SOM cells
# plt.rc('font', size=8)
# for i, lid in enumerate(lids):
#     fig, ax = plt.subplots()
#     m = np.load(os.path.join(path, str(lid), "m.npy"))
#     # p_t_s = som.predict_t_s_model(som.distance_matrix, m, som.term_size[lid])
#     x = som.distance_matrix
#     diff_x = m[:, :, None, som.term_size[lid]:] - x[None, None, :]
#     dist_x = np.linalg.norm(diff_x, axis=-1).reshape((-1, len(x)))
#     bmu_idx = np.argmin(dist_x, axis=1)
#     bmu_x = WCS_CHIPS[bmu_idx].reshape((m.shape[0], m.shape[1], -1))
#     mu_w = lab2rgb(bmu_x)
#     plt.imshow(mu_w, interpolation='nearest')
#     words = m[..., :som.term_size[lid]].argmax(-1)
#     words = np.apply_along_axis(lambda x: [som.word_map[lid][i] for i in x], 0, words)
#     x, y = np.arange(m.shape[0])-0.15, np.arange(m.shape[1]) + 0.15
#     for i, xx in enumerate(x):
#         for j, yy in enumerate(y):
#             plt.text(xx, yy, words[i, j])
#     if i == 0:
#         ax.set_ylabel("SOM")
#     fig.tight_layout()
#     fig.savefig(os.path.join(path, f"grid_cells_{lid}.pdf"))
# plt.rc('font', size=16)

# # Plot all selected LIDs on one plot without animation
# handles = []
# labels = []
# fig, ax = plt.subplots()
# for i, (lid, scores) in enumerate(scores_dict.items()):
#     if lid not in lids: continue
#
#     ax.set_xlabel(r"Complexity; $I(W, C)$ bits")
#     ax.set_ylabel("Information Loss; KL-Divergence bits")
#     ax.set_title(lang_strs.loc[lid, "language"])
#
#     frontier = pickle.load(open(f"frontier/learnability_languages/{lid}.p", "rb"))
#     ax.plot(frontier[0], frontier[1])
#
#     X, Y = scores[:-1, :2].T
#     U, V = np.diff(scores[:, :2], axis=0).T
#     ax.quiver(X, Y, U, V,
#               angles="xy",
#               scale_units="xy",
#               scale=1,
#               width=0.005,
#               headwidth=2,
#               color=colors[i % len(colors)],
#               )
#     ax.scatter(
#         X, Y, s=6, edgecolor="white", linewidth=0.5, color=colors[i % len(colors)]
#     )
#     plt.plot()
#     handles.append(Line2D([], [],
#                           color="white",
#                           markerfacecolor=colors[i % len(colors)],
#                           marker="o",
#                           markersize=10))
#     labels.append(f"{lang_strs.loc[lid, 'language']} ({som.term_size[lid]})")
# plt.legend(handles, labels)
# fig.tight_layout()
# fig.savefig(os.path.join(path, "learning_trajectories.pdf"))
# plt.show()
#
# Plot accuracy
for i, lid in enumerate(lids):
    plt.figure()
    results = pickle.load(open("accuracies.pkl", "rb"))
    n_conv = pickle.load(open("n_conv.pkl", "rb"))
    plt.ylim((0, 1))
    means = np.array([np.mean(vals) for _, vals in sorted(results[f"lang_{lid}"].items(), key=lambda x: x[0])])
    stds = np.array([np.std(vals) for _, vals in sorted(results[f"lang_{lid}"].items(), key=lambda x: x[0])])
    plt.plot(sample_range, means, color=colors[i % len(colors)])
    plt.fill_between(sample_range, means - stds, means + stds, color=colors[i % len(colors)], alpha=0.33)

    smeans = np.array([np.mean(vals) for _, vals in sorted(results[f"suboptimal_{lid}"].items(), key=lambda x: x[0])])
    sstds = np.array([np.std(vals) for _, vals in sorted(results[f"suboptimal_{lid}"].items(), key=lambda x: x[0])])
    plt.plot(sample_range, smeans, color="grey")
    plt.fill_between(sample_range, smeans - sstds, smeans + sstds, color="grey", alpha=0.33)
    plt.xlabel("Number of samples")
    if i == 0:
        plt.ylabel("Accuracy")
    plt.xlim([0, 20000])
    plt.gcf().tight_layout()
    plt.savefig(os.path.join(path, f"accuracy_{lid}.pdf"))
    plt.show()

# Plot mode maps for the distributions
for i, lid in enumerate(lids):
    p_t_s = np.load(os.path.join(path, str(lid), "50000_pt_s_all.npy")).mean(0)
    p_t_s_som = som.pts[lid]
    s_p_t_s = np.load(os.path.join(spath, str(lid), "50000_pt_s_all.npy")).mean(0)
    ps = 0.1 * som.ps_universal + 0.9 * som.ps[lid]

    fig, axes = plt.subplots(3, 1, figsize=(5, 5))
    ax = axes[0]
    mode_map(p_t_s_som.T, ps, ax=ax)
    if i == 0:
        ax.set_ylabel("Data")

    ax = axes[1]
    mode_map(p_t_s, ps, ax=ax)
    if i == 0:
        ax.set_ylabel("SOM")

    ax = axes[2]
    mode_map(s_p_t_s, ps[rotation[lid]["rotation_indices"], :], ax=ax)
    if i == 0:
        ax.set_ylabel("Hypothetical")

    fig.tight_layout()
    fig.savefig(os.path.join(path, f"mode_map_comparison_{lid}.pdf"))
    plt.show()

# Plot averaged learning trajectories
for i, lid in enumerate(lids):
    scores = scores_dict[lid]
    sscores = sscores_dict[lid]

    handles = []
    labels = []

    fig, ax = plt.subplots()

    ax.set_xlabel("Complexity; $I(W, C)$ bits")
    ax.set_ylabel("Information Loss; KL-Divergence")
    ax.set_title(lang_strs.loc[lid, "language"] + rf"; $|W|={som.term_size[lid]}$")

    # frontier = pickle.load(open(f"frontier/learnability_languages/{lid}.p", "rb"))
    # ax.plot(frontier[0], frontier[1], color=colors[i % len(colors)])

    X, Y = scores[:-1, :2].T
    Xs, Ys = sscores[:-1, :2].T
    Xerr, Yerr = scores[:-1, 2:].T
    U, V = np.diff(scores[:, :2], axis=0).T
    Us, Vs = np.diff(sscores[:, :2], axis=0).T

    if not animate_traj:
        ax.quiver(X, Y, U, V,
                  angles="xy",
                  scale_units="xy",
                  scale=1,
                  width=0.01,
                  headwidth=2,
                  color=colors[i % len(colors)],
                  )
        ax.scatter(
            X, Y, s=12, edgecolor="white", linewidth=0.5, color=colors[i % len(colors)]
        )
        handles.append(Line2D([], [],
                              color="white",
                              markerfacecolor=colors[i % len(colors)],
                              marker="o",
                              markersize=10))
        labels.append(f"Data")
        ax.quiver(Xs, Ys, Us, Vs,
                  angles="xy",
                  scale_units="xy",
                  scale=1,
                  width=0.01,
                  headwidth=2,
                  color="grey",
                  )
        ax.scatter(
            Xs, Ys, s=12, edgecolor="white", linewidth=0.5, color="grey"
        )
        handles.append(Line2D([], [],
                              color="white",
                              markerfacecolor="grey",
                              marker="o",
                              markersize=10))
        labels.append(f"Hypothetical")
        # ax.fill_between(X, Y-Yerr, Y+Yerr, color=colors[i % len(colors)], alpha=0.33)
        # ax.errorbar(X, Y, xerr=Xerr, yerr=Yerr, fmt="none", alpha=0.45,
        #             ecolor='k', capthick=0.1, elinewidth=0.1)  # Used fifty averaging runs
        # for j, n in enumerate(sample_range[:-1]):
        #     plt.text(X[j], Y[j], n)
        ax.set_xlim([-0.1, 2.2])

        if lid == 2:
            x1, x2, y1, y2 = -0.005, 0.025, 32.2, 33.5
            zoom = 16
        elif lid == 32:
            x1, x2, y1, y2 = -0.000, 0.017, 33.6, 34.4
            zoom = 26
        elif lid == 35:
            x1, x2, y1, y2 = -0.005, 0.020, 33.5, 35
            zoom = 16
        else:
            x1, x2, y1, y2 = -0.005, 0.025, 34, 34.8
            zoom = 20
        axins = zoomed_inset_axes(ax, zoom=zoom, loc='upper right')
        axins.quiver(X, Y, U, V,
                     angles="xy",
                     scale_units="xy",
                     scale=1,
                     width=0.05,
                     headwidth=2,
                     color=colors[i % len(colors)],
                     )
        axins.scatter(
            X, Y, s=6, edgecolor="white", linewidth=0.5, color=colors[i % len(colors)]
        )
        # axins.quiver(Xs, Ys, Us, Vs,
        #           angles="xy",
        #           scale_units="xy",
        #           scale=1,
        #           width=0.01,
        #           headwidth=2,
        #           color="grey",
        #           )
        # axins.scatter(
        #     Xs, Ys, s=12, edgecolor="white", linewidth=0.5, color="grey"
        # )
        if lid in [2, 32, 35, 108]:
            # sub region of the original image
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.yaxis.get_major_locator().set_params(nbins=7)
            axins.xaxis.get_major_locator().set_params(nbins=7)
            axins.tick_params(labelleft=False, labelbottom=False)
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

        if i == 0:
            ax.legend(handles, labels, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"learning_traj_{lid}.pdf"))
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
        anim.save(os.path.join(path, f"learning_traj_{lid}.gif"), dpi=300)

    # plt.show()

# Plot mode maps
if animate:
    for i, (lid, scores) in enumerate(scores_dict.items()):
        if lid not in lids: continue

        if not os.path.exists(os.path.join(path, str(lid), f"mode_maps")):
            os.mkdir(os.path.join(path, str(lid), f"mode_maps"))
        pt_s_arr = []
        for file in glob.glob(os.path.join(path, str(lid), "*.npy")):
            pt_s = np.load(file)
            n_samples = int(file.split(os.sep)[-1].split("_")[0])
            pt_s_arr.append((n_samples, pt_s))
        mode_maps = []
        for j, (n_samples, pt_s) in enumerate(sorted(pt_s_arr, key=lambda x: x[0])):
            mode_map(pt_s.mean(0))
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
            fig.savefig(os.path.join(path, str(lid), "mode_maps", f"{j:03}.jpg"))

        fp_in = os.path.join(path, f"{lid}", "mode_maps/*.jpg")
        fp_out = os.path.join(path, f"{lid}", f"mode_map_{lid}.gif")
        images = list(map(lambda filename: imageio.imread(filename), sorted(glob.glob(fp_in))))
        imageio.mimsave(os.path.join(fp_out), images, duration=0.25)
