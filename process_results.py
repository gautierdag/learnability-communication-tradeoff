import os
import sys
import numpy as np
import pandas as pd
import glob

from convergence import get_accuracy, n_sample_converged
from noga.tools import DKL, MI
from som import SelfOrganisingMap, sample_range


def score(arr, pts, ps):
    infls = []
    mis = []
    for pt_s_h in arr:
        pst_h = pt_s_h * ps
        inf_loss = DKL(pst_h.T, pts)
        mutual_info_h = MI(pst_h)
        infls.append(inf_loss)
        mis.append(mutual_info_h)
    return infls, mis


def accuracy(arr, lid, som, language):
    accs = []
    for p_t_s in arr:
        accs.append(get_accuracy(p_t_s, lid, som, language))
    return accs


def convergence(arr, window=3, threshold=1):
    convs = []
    for accuracies in arr:
        for i in np.arange(len(accuracies) - window):
            w = np.array([acc for j, acc in enumerate(accuracies) if i <= j < i + window])
            error = w - w.mean()
            rmsd = np.sqrt(np.dot(error, error) / (len(w) - 1))
            if rmsd < threshold:
                convs.append(sample_range[i])
                break
        else:
            convs.append(sample_range[i])
    return convs


if __name__ == '__main__':
    seed = 42
    path = os.path.join("output", "som", f"{seed}")
    lid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if not os.path.exists(os.path.join(path, "processed")):
        os.mkdir(os.path.join(path, "processed"))

    som = SelfOrganisingMap()
    data = som.term_data
    data = data[~pd.isna(data["word"])]

    print(f"Processing Language {lid}")
    lid_folder = os.path.join(path, str(lid))
    results = pd.DataFrame()

    ps = 0.9 * som.ps[lid] + 0.1 * som.ps_universal
    pts = som.pts[lid]
    data = data[data["language"] == lid]

    samples = sorted(glob.glob(os.path.join(lid_folder, "*_all.npy")),
                     key=lambda x: int(x.split(os.sep)[-1].split("_")[0]))
    for i, sample_file in enumerate(samples):
        print(f"Processing {sample_file}")

        n_samples = int(sample_file.split(os.sep)[-1].split("_")[0])
        p_t_s_arr = np.load(sample_file)
        average_k = len(p_t_s_arr)
        results_dict = {}

        # Calculate scores
        inf_losses, mutual_infs = score(p_t_s_arr, pts, ps)
        results_dict.update({
            "information_loss": inf_losses,
            "mutual_information": mutual_infs,
        })

        # Calculate convergence
        accs = accuracy(p_t_s_arr, lid, som, data)
        results_dict.update({"accuracy": accs})

        results_dict.update({
            "language": [lid] * average_k,
            "n_samples": [n_samples] * average_k,
            "average_k": list(range(average_k))
        })

        results = results.append(pd.DataFrame.from_dict(results_dict), ignore_index=True)

    # Get point of convergence
    n_conv = convergence(results["accuracy"].to_numpy().reshape(average_k, -1))
    results["n_convergence"] = n_conv * (len(results) // len(n_conv))
    results.to_csv(os.path.join(path, "processed", f"{lid}.csv"))
