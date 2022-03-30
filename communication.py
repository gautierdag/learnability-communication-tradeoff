import argparse
import os
import pickle
from typing import Tuple
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.stats import multivariate_normal
import glob
import multiprocessing

from learnability_frontier import fit_optimal_curve
from som import SelfOrganisingMap, sample_range

NUM_MEANINGS = 330


class MutualInfoCalculator(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def read_language_data(fpath: str = "wcs/term.txt") -> pd.DataFrame:
        df = pd.read_csv(fpath, delimiter="\t", header=None)
        df.columns = ["language", "speaker", "chip", "word"]
        return df

    @staticmethod
    def read_colour_data(fpath: str = "wcs/cnum-vhcm-lab-new.txt") -> pd.DataFrame:
        df = pd.read_csv(fpath, sep="\t", header=0, index_col="#cnum")
        df = df.sort_values(by="#cnum")[["L*", "a*", "b*"]].copy().reset_index()
        return df

    @staticmethod
    def get_px(fpath: str = "wcs/term.txt") -> ArrayLike:
        """
        Gets p_x from the language data by using the frequency of each word for
        each chip and averaging over all languages.
        Args:
            fpath: path to the language data file
        Returns:
            p_x: probability of each meaning
        """
        df = pd.read_csv(fpath, delimiter="\t", header=None)
        df.columns = ["language", "speaker", "chip", "word"]

        # frequentist probability of a chip given word and language
        per_word_count_df = df.groupby(["language", "word", "chip"]).speaker.agg(
            individual_count_per_chip_per_word="count"
        )
        total_chip_count_df = df.groupby(["language", "word"]).chip.agg(
            total_chips_per_word="count"
        )
        p_chip_word_language = (
                per_word_count_df["individual_count_per_chip_per_word"]
                / total_chip_count_df["total_chips_per_word"]
        )

        # frequentist probability of a word given chip and language
        per_chip_count_df = df.groupby(["language", "chip", "word"]).speaker.agg(
            individual_count_per_word_per_chip="count"
        )
        total_word_count_df = df.groupby(["language", "chip"]).word.agg(
            total_words_per_chip="count"
        )
        p_word_chip_language = (
                per_chip_count_df["individual_count_per_word_per_chip"]
                / total_word_count_df["total_words_per_chip"]
        )

        p_xs = []
        for language in df.language.unique():
            # convert words to indices
            words = p_chip_word_language[language].reset_index().word.unique()
            word_to_index = {word: i for word, i in zip(words, range(len(words)))}
            # index_to_word =  {i: word for word, i in zip(words, range(len(words))) }

            # assigne a word to each chip
            chip_to_word = np.zeros(NUM_MEANINGS)
            try:
                # Find the most common word for each chip and assign chip to that word\n
                for chip_num, word in (
                        p_word_chip_language[language].groupby(level=0).idxmax().values
                ):
                    chip_to_word[chip_num - 1] = word_to_index[word]
            except TypeError as e:
                print(f"TypeError: missing chip data for language {language}")
                continue

            # Find the frequency of that color in data, then split it equally across all the chips assigned to it\n
            word_frequencies = (
                p_chip_word_language[language]
                    .reset_index()
                    .word.apply(lambda x: word_to_index[x])
                    .value_counts()
            )

            p_x = np.zeros(NUM_MEANINGS)
            for i in range(NUM_MEANINGS):
                p_x[i] = 1 / word_frequencies[chip_to_word[i]]

            p_xs.append(p_x)

        # average prior over all languages
        p_x = np.array(p_xs).mean(axis=0)

        return p_x

    @staticmethod
    def get_pxGy(fpath: str = "wcs/cnum-vhcm-lab-new.txt", covariance=64) -> ArrayLike:
        """
        Args:
            fpath: path to the colour data file
            covariance: covariance of the colour space (@TODO: This is a hyperparameter than needs to be justified)
        Returns:
            p_xGy
        """
        chip2lab = pd.read_csv(fpath, sep="\t", header=0, index_col="#cnum")
        chip2lab = (
            chip2lab.sort_values(by="#cnum")[["L*", "a*", "b*"]].copy().reset_index()
        )
        labspace = chip2lab[["L*", "a*", "b*"]].values

        p_xGy = np.zeros((NUM_MEANINGS, NUM_MEANINGS))
        for i in range(NUM_MEANINGS):
            # the multivariate pdf over the lab space dimension to go back into chip/meaning space
            p_xGy[i] = multivariate_normal.pdf(
                labspace, mean=labspace[i], cov=covariance
            )

        p_xGy = p_xGy / p_xGy.sum(axis=1, keepdims=True)

        return p_xGy

    def get_MI(
            self, flan: str = "wcs/term.txt", fclab: str = "wcs/cnum-vhcm-lab-new.txt"
    ) -> float:
        p_x = self.get_px(flan)
        p_xGy = self.get_pxGy(fclab)

        p_xy = p_xGy * p_x[:, np.newaxis]
        p_xy = p_xy / np.sum(p_xy)

        mi = np.nansum(
            p_xy
            * np.log2(
                p_xy
                / (p_xy.sum(axis=0, keepdims=True) * p_xy.sum(axis=1, keepdims=True))
            )
        )

        return mi


class LanguageSampler:
    def __init__(self, fpath: str = None) -> None:
        super().__init__()
        self.fpath = fpath
        with open(self.fpath, "rb") as f:
            self.prob_matrix = np.load(f)
        self.num_words = self.prob_matrix.shape[1]
        assert self.prob_matrix.shape[0] == 330, "q matrix must be 330xW"
        assert np.isclose(self.prob_matrix.sum(), 1), "q matrix sum must = 1"

    def sample_indices(self, n: int) -> Tuple[ArrayLike, ArrayLike]:
        # Create a flat copy of the 2D joint distribution
        flat = self.prob_matrix.flatten()

        # Then, sample an index from the 1D array with the
        # probability distribution from the original array
        sample_index = np.random.choice(a=flat.size, p=flat, size=n, replace=True)

        # Take this index and adjust it so it matches the original array
        chip_indices, word_indices = np.unravel_index(
            sample_index, self.prob_matrix.shape
        )

        return chip_indices, word_indices


def func(args):
    sampler_, sample_range_, nw, pts, path_ = args
    c_, w_ = sampler_.sample_indices(sample_range_[-1])
    som_ = SelfOrganisingMap()
    m_ = np.zeros((som_.size, som_.size, nw + som_.distance_matrix.shape[0]))
    score_ = som_.learn_language_from_samples(None, (w_, c_), sample_range_,
                                              nw, m_, pts.T,
                                              save_pt_s=path_)
    return score_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOM on CE-optimal languages")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--average_k", type=int, default=10,
                        help="The number of learners to average over for the developmental plots.")
    parser.add_argument("--workers", type=int, default=None,
                        help="If given, then use multiprocessing with given number of workers.")
    parser.add_argument("--optimal", action="store_true", default=False,
                        help="Whether to use CE-optimal or suboptimal languages.")
    parser.add_argument("--i", type=int, default=None, help="Select a particular beta by index. ")

    args = parser.parse_args()

    print(args)

    seed = args.seed
    average_k = args.average_k
    workers = args.workers
    save_scores = False
    np.random.seed(seed)

    if args.optimal:
        files = glob.glob(os.path.join("frontier", "q_matrices", "*"))
    else:
        files = glob.glob(os.path.join("output", "worst_qs", "*"))

    betas = {}
    num_words = {}
    rates = []
    distortions = []
    scores = {}

    prev_num_words = 0
    for i, f in enumerate(files):
        beta, rate, distortion = f.split(os.sep)[-1].split("_")
        distortion = distortion.split(".npy")[0]
        beta, rate, distortion = float(beta), float(rate), float(distortion)
        betas[i] = beta
        rates.append(rate)
        distortions.append(distortion)
        sampler = LanguageSampler(f)
        num_words[i] = sampler.num_words

        if num_words[i] <= prev_num_words:
            continue
        prev_num_words = num_words[i]

        if args.i is not None and i != args.i:
            continue

        print(i, f, num_words[i])

        som = SelfOrganisingMap()

        save_path = os.path.join("output", "som", f"{seed}", "ce", f"{int(beta) if not args.optimal else beta}")
        if not os.path.exists(os.path.join("output", "som", f"{seed}", "ce")):
            os.mkdir(os.path.join("output", "som", f"{seed}", "ce"))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        sampling_scores = []
        if workers is not None:
            with multiprocessing.Pool(processes=workers) as p:
                sampling_scores = p.map(func, [(sampler, sample_range, sampler.num_words,
                                                sampler.prob_matrix, save_path)
                                               for j in range(average_k)])
        else:
            for k in range(average_k):
                # c and w are the chip and word indices as arrays of size N
                c, w = sampler.sample_indices(sample_range[-1])

                m = np.zeros((som.size, som.size, sampler.num_words + som.distance_matrix.shape[0]))
                sampling_scores.append(som.learn_language_from_samples(None, (w, c), sample_range,
                                                                       sampler.num_words, m, sampler.prob_matrix.T,
                                                                       save_pt_s=save_path))

                # Load all saved p_t_s and join to already calculated ones
                if save_path:
                    for s in sample_range:
                        p_t_s = np.load(os.path.join(save_path, f"{s}_pt_s.npy"))
                        if not os.path.exists(os.path.join(save_path, f"{s}_pt_s_all.npy")):
                            joined = p_t_s[None, :]
                        else:
                            joined = np.load(os.path.join(save_path, f"{s}_pt_s_all.npy"))
                            joined = np.vstack([joined, p_t_s[None, :]])
                        np.save(os.path.join(save_path, f"{s}_pt_s_all.npy"), joined)

        np_scores = np.array(sampling_scores)
        scores[i] = np.hstack([np.mean(np_scores, 0), np.std(np_scores, 0)])

    if save_scores:
        if args.optimal:
            pickle.dump((betas, num_words, scores), open(f"output/som/{seed}/ce/optimal_scores_dict.p", "wb"))
        else:
            pickle.dump((betas, num_words, scores), open(f"output/som/{seed}/ce/suboptimal_scores_dict.p", "wb"))
