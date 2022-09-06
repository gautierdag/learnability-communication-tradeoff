import os
import pickle
import sys
import numpy as np
import pandas as pd
import glob

from communication import LanguageSampler
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


def convergence(arr, window=20, threshold=0.005):
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
            convs.append(sample_range[-1])
    return convs


if __name__ == '__main__':
    seed = 42
    ce = False
    suboptimal = False
    rotations = pickle.load(open("pickle/worst_qs_rotated.p", "rb"))
    data_dir = "ablations"
    if not ce:
        if suboptimal:
            path = os.path.join("output", "som", f"{seed}", "suboptimal")
        else:
            path = os.path.join("output", "som", f"{seed}", data_dir)
        lid = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        if not os.path.exists(os.path.join(path, "processed")):
            os.mkdir(os.path.join(path, "processed"))

        som = SelfOrganisingMap(subopt=suboptimal,
                                wcs_path="wcs_en" if data_dir == "en" else "wcs",
                                sampling="english" if data_dir == "en" else "corpus")

        print(f"Processing Language {lid}")
        lid_folder = os.path.join(path, str(lid))
        results = pd.DataFrame()

        ps = 0.9 * som.ps[lid] + 0.1 * som.ps_universal
        if suboptimal:
            ps = ps[rotations[lid]["rotation_indices"], :]
        pts = som.pts[lid]

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

            results_dict.update({
                "language": [lid] * average_k,
                "n_samples": [n_samples] * average_k,
                "average_k": list(range(average_k))
            })

            results = results.append(pd.DataFrame.from_dict(results_dict), ignore_index=True)
        results.to_csv(os.path.join(path, "processed", f"{lid}.csv"))
    else:
        path = os.path.join("output", "som", f"{seed}", "ce")
        # folders = sorted(glob.glob(os.path.join(path, "*")), key=lambda x: float(x.split(os.sep)[-1]))
        betas = [1.0942937012608183, 1.2226402776921537, 1.4439291955225915, 1.7171308728756145, 11.79415373832906] #[float(x.split(os.sep)[-1]) for x in folders]
        beta = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        beta = betas[beta]

        som = SelfOrganisingMap()

        print(f"Processing Language {beta}")
        lid_folder = os.path.join(path, str(beta))
        results = pd.DataFrame()

        mx = glob.glob(os.path.join("frontier", "q_matrices", f"{beta}*"))[0]
        sampler = LanguageSampler(mx)

        ps = som.ps_universal
        pts = sampler.prob_matrix.T

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

            results_dict.update({
                "language": [beta] * average_k,
                "n_samples": [n_samples] * average_k,
                "average_k": list(range(average_k))
            })

            results = results.append(pd.DataFrame.from_dict(results_dict), ignore_index=True)

        if not os.path.exists(os.path.join(path, "processed")):
            os.mkdir(os.path.join(path, "processed"))
        results.to_csv(os.path.join(path, "processed", f"{beta}.csv"))

    scores_dict = {}
    for f in glob.glob(f"output/som/42/{data_dir}/processed/*.csv"):
        results = pd.read_csv(f, index_col=0)
        beta = float(f.split(os.sep)[-1][:-4])
        arr = []
        means = results.groupby("n_samples").mean()[["mutual_information", "information_loss"]].to_numpy()
        stds = results.groupby("n_samples").std()[["mutual_information", "information_loss"]].to_numpy()
        scores_dict[beta] = np.hstack([means, stds])
    pickle.dump(scores_dict, open(f"output/som/42/{data_dir}/scores_dict.p", "wb"))