import argparse
import glob
import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

from process_results import score
from som import SelfOrganisingMap, sample_range
from noga.figures import mode_map

parser = argparse.ArgumentParser(description="Run SOM on WCS data")
parser.add_argument("--average_k", type=int, default=5, help="The number of learners to "
                                                             "average over for the developmental plots.")
parser.add_argument("--lid", type=int, default=None, help="ID of language to learn.")

args = parser.parse_args()
print(args)

seed = 42
lid = args.lid  # Needed for some to run but irrelevant as long as integer in [1, 110]
som = SelfOrganisingMap()
M = 2  # Number of grayscale terms
N = som.term_size[lid] - M  # Number of terms to use

print(f"Generating random language with {N + M} terms.")

random.seed(seed)
np.seterr(divide="ignore")
np.random.seed(seed)

grid = np.zeros((8, 40), dtype=int) - 1  # The Munsell chip-grid without the 10 graysale chips

shuffle = np.arange(grid.size)
np.random.shuffle(shuffle)

idx = np.unravel_index(shuffle[:N], grid.shape)
grid[idx] = np.arange(N)

# Randomly partition main grid
while np.any(grid == -1):
    for i in shuffle:
        idx = np.unravel_index(i, grid.shape)
        if grid[idx] == -1:
            u, v = idx
            subset = grid[max(0, u - 1):min(u + 2, grid.shape[0]), max(0, v - 1):min(v + 2, grid.shape[1])]
            if np.all(subset == -1):
                continue
            vals, counts = np.unique(subset, return_counts=True)
            vals, counts = vals[1:], counts[1:]  # Drop counts for -1
            probs = counts / counts.sum()
            grid[idx] = np.random.choice(vals, 1, p=probs)

splits = np.sort(np.random.choice(np.arange(10), M - 1, replace=False))
splits = np.append(splits, 10)
grayscale = np.zeros(10, dtype=int)
start = 0
for t, end in enumerate(splits, N):
    grayscale[start:end] = t
    start = end

# Convert assignments to probability distributions
chip_data = pd.read_csv(
    os.path.join("wcs", "chip.txt"),
    sep="\t",
    index_col=0,
    names=["row", "col", "pos"],
)
chip_data["nrow"] = chip_data["row"].apply(lambda x: "ABCDEFGHIJ".index(x))
flat = np.zeros(grid.size + grayscale.size, dtype=int) - 1

grid_data = chip_data[chip_data["col"] != 0]
grid_idxs = 40 * (grid_data["nrow"] - 1) + grid_data["col"]
flat[grid_idxs.index - 1] = grid.flatten()[grid_idxs - 1]

grayscale_data = chip_data[chip_data["col"] == 0]
grayscale_data = grayscale_data.sort_values("row")
flat[grayscale_data.index - 1] = grayscale

pc_w = np.zeros((N+M, flat.size))
for c, t in enumerate(flat):
    pc_w[t, c] = 1.0
pwc = pc_w / pc_w.sum()

# mode_map(pc_w.T)
# plt.show()

# Global parameters
n = sample_range[-1]
average_k = args.average_k
wcs = "wcs"
sampling = "corpus"
prior = "capacity" if wcs != "wcs_en" else "english"
term_dist = "corpus" if wcs != "wcs_en" else "english"
save_xling = True  # Whether to save the cross-linguistic feature space
grid_search = False
save_p = True
save_samples = False

if not os.path.exists("output"):
    os.mkdir("output")
if not os.path.exists("output/som"):
    os.mkdir("output/som")
if not os.path.exists(f"output/som/{seed}"):
    os.mkdir(f"output/som/{seed}")
if not os.path.exists(f"output/som/{seed}/random"):
    os.mkdir(f"output/som/{seed}/random")
for lid in [lid]:
    if not os.path.exists(f"output/som/{seed}/random/{lid}"):
        os.mkdir(f"output/som/{seed}/random/{lid}")

som_args = {
    "wcs_path": wcs,
    "sampling": sampling,
    "color_prior": prior,
    "term_dist": term_dist,
    # "size": 2,
    # "alpha": 1e-4,
    "model_init": "zeros"
}
print(som_args)

scores = []

pickle.dump(pwc, open(f"output/som/{seed}/random/{lid}/pts.p", "wb"))

for k in trange(average_k):
    som = SelfOrganisingMap(**som_args)
    som.term_size[lid] = pwc.shape[0]
    som.pts[lid] = pwc
    som.models[lid] = np.zeros((som.size, som.size, pwc.shape[0] + som.distance_matrix.shape[0]))

    index_matrix = np.arange(pwc.size)
    samples = tuple(
        np.unravel_index(
            np.random.choice(index_matrix, n, p=pwc.flatten()), pwc.shape
        )
    )
    m = np.zeros((som.size, som.size, pwc.shape[0] + som.distance_matrix.shape[0]))
    language_scores = som.learn_language_from_samples(
        None, samples, sample_range, pwc.shape[0], m, pwc,
        os.path.join(f"output/som/{seed}/random", f"{lid}") if save_p else None)

    # Load all saved p_t_s and join to already calculated ones
    if save_p:
        for s in sample_range:
            p_t_s = np.load(f"output/som/{seed}/random/{lid}/{s}_pt_s.npy")
            if not os.path.exists(f"output/som/{seed}/random/{lid}/{s}_pt_s_all.npy"):
                joined = p_t_s[None, :]
            else:
                joined = np.load(f"output/som/{seed}/random/{lid}/{s}_pt_s_all.npy")
                joined = np.vstack([joined, p_t_s[None, :]])
            np.save(f"output/som/{seed}/random/{lid}/{s}_pt_s_all.npy", joined)

# Process results
path = os.path.join("output", "som", f"{seed}", "random")
if not os.path.exists(os.path.join(path, "processed")):
    os.mkdir(os.path.join(path, "processed"))

print(f"Processing Language {lid}")
lid_folder = os.path.join(path, str(lid))
results = pd.DataFrame()

ps = som.ps_universal
samples = sorted(glob.glob(os.path.join(lid_folder, "*_all.npy")),
                 key=lambda x: int(x.split(os.sep)[-1].split("_")[0]))
for i, sample_file in enumerate(samples):
    print(f"Processing {sample_file}")

    n_samples = int(sample_file.split(os.sep)[-1].split("_")[0])
    p_t_s_arr = np.load(sample_file)
    average_k = len(p_t_s_arr)
    results_dict = {}

    # Calculate scores
    inf_losses, mutual_infs = score(p_t_s_arr, pwc, ps)
    results_dict.update({
        "information_loss": inf_losses,
        "mutual_information": mutual_infs,
    })

    results_dict.update({
        "language": [lid] * average_k,
        "n_samples": [n_samples] * average_k,
        "average_k": list(range(average_k))
    })

    results = results.append(pd.DataFrame.from_dict(results_dict), ignore_index=True)
results.to_csv(os.path.join(path, "processed", f"{lid}.csv"))

scores_dict = {}
for f in glob.glob(f"output/som/42/random/processed/*.csv"):
    results = pd.read_csv(f, index_col=0)
    beta = float(f.split(os.sep)[-1][:-4])
    arr = []
    means = results.groupby("n_samples").mean()[["mutual_information", "information_loss"]].to_numpy()
    stds = results.groupby("n_samples").std()[["mutual_information", "information_loss"]].to_numpy()
    scores_dict[beta] = np.hstack([means, stds])
pickle.dump(scores_dict, open(f"output/som/42/random/scores_dict.p", "wb"))
