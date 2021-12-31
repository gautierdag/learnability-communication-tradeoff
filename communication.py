from typing import Tuple
from typing import Tuple, List
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from itertools import product
from string import ascii_uppercase
from scipy.stats import multivariate_normal


NUM_MEANINGS = 330


class LanguageSampler(object):
    def __init__(self, fpath: str = None) -> None:
        super().__init__()
        self.fpath = fpath
        self.prob_matrix = None
        self.read_prob_matrix()

    def read_prob_matrix(self) -> ArrayLike:
        with open(self.fpath, "rb") as f:
            self.prob_matrix = np.load(f)

    def sample_index(self) -> Tuple:
        word_idx = np.random.choice(
            self.prob_matrix.shape[0], replace=True, p=self.prob_matrix.sum(axis=1)
        )
        chip_idx = np.random.choice(
            self.prob_matrix.shape[1], replace=True, p=self.prob_matrix.sum(axis=0)
        )
        return word_idx, chip_idx

    def sample_language(self, N: int = 1) -> Tuple[List, List]:
        idx2colour = {idx: idx for idx in range(330)}
        words = ["".join(i) for i in product(ascii_uppercase, repeat=2)]
        idx2word = {idx: word for idx, word in enumerate(words)}

        word_list = []
        colour_list = []
        for _ in range(N):
            word_idx, chip_idx = self.sample_index()
            word = idx2word[word_idx]
            colour = idx2colour[chip_idx]

            word_list.append(word)
            colour_list.append(colour)

        return word_list, colour_list

    @staticmethod
    def output_language_file(
        words: list,
        colours: list,
        opath: str = "./lan.txt",
        lan_idx: int = 1,
        spk_idx: int = 1,
    ) -> None:
        with open(opath, "a") as f:
            for word, colour in zip(words, colours):
                print(
                    str(lan_idx)
                    + "\t"
                    + str(spk_idx)
                    + "\t"
                    + str(colour)
                    + "\t"
                    + word,
                    file=f,
                )


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
            words = p_chip_word_language[1].reset_index().word.unique()
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
    def get_pxGy(fpath: str = "wcs/cnum-vhcm-lab-new.txt", covariance=100) -> ArrayLike:
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

    def get_MI(self, flan: str = None, fclab: str = None) -> float:
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


if __name__ == "__main__":
    sampler = LanguageSampler("./tmp_test.npy")
    sample_words, sample_colours = sampler.sample_language(N=10)
    sampler.output_language_file(sample_words, sample_colours)
    sample_words, sample_colours = sampler.sample_language(N=20)
    sampler.output_language_file(sample_words, sample_colours, lan_idx=2, spk_idx=2)
