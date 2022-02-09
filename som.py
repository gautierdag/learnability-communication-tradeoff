import itertools
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learnability import GaussianLanguageModel
from noga.figures import mode_map


NUM_CHIPS = 330


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_xling_features(data: pd.DataFrame, save: bool = True) -> np.ndarray:
    """ Calculate the cross linguistic feature space for each colour chip. """
    if os.path.exists("xling.p"):
        return pickle.load(open("xling.p", "rb"))

    mat = []
    for lang_id, lang in data.groupby("language"):
        t = lang["word"].nunique()
        t_unique = lang["word"].dropna().unique()
        p_t_s = np.zeros((NUM_CHIPS, t))
        for i, (chip, chip_data) in enumerate(lang.groupby("chip")):
            c_freq = chip_data.groupby("word").size().reindex(t_unique, fill_value=0)
            p_t_s[i, :] = c_freq / chip_data.shape[0]
        mat.append(p_t_s)
    mat = np.hstack(mat)

    if save:
        pickle.dump(mat, open("xling.p", "wb"))

    return mat


def get_distance_matrix(data: np.ndarray):
    dist = np.zeros((NUM_CHIPS, NUM_CHIPS))
    for i, row in enumerate(data):
        for j, col in enumerate(data[i:], i):
            d = np.linalg.norm(row - col)
            dist[i, j] = d
    dist += dist.T
    return dist


class SelfOrganisingMap:
    """ Define a self-organising map"""
    def __init__(self, size: int, term_size: int, alpha: float, sigma: float, term_weight: float,
                 sem_size: int = 3, features: str = "perc", sampling: str = "corpus"):
        """ Initialise a new self-organising map.

        Args:
            size: The size of the square matrix M
            term_size: The number of term features
            sem_size: The number of semantic features
            alpha: The learning rate
            sigma: Neighbourhood radius
            term_weight: Term importance weight
            features: Type of semantic features to use. Either 'perc' or 'xling'
            sampling: Estimation method for p(t). Either unif or rel_freq
        """
        assert size > 0, "Matrix size must be larger than 0."
        assert term_size > 0, "Features size must be larger than 0."
        assert 0 < alpha < 1, "Learning rate alpha must be between 0 and 1."
        assert 0 < sigma, "Neighbourhood radius must be positive."
        assert 0 < term_weight <= 1, "Term importance must be in (0,1]."
        assert features in ["perc", "xling"], "Invalid feature type given"
        assert sampling in ["corpus", "uniform"], "Invalid estimation type for p(t)"

        self.m = size
        self.k = term_size
        self.d = sem_size
        self.M = np.zeros((self.m, self.m, self.k + NUM_CHIPS))  # Semantic feature space from D-matrix with size 330
        self.alpha = alpha
        self.sigma = sigma
        self.a = term_weight
        self.features = features
        self.p_t_estimator = sampling

        # Map coordinate array for calculating d_map
        map_idx = np.arange(0, self.m ** 2).reshape((self.m, self.m))
        map_idx = np.stack(np.unravel_index(map_idx, (self.m, self.m)), axis=2)
        self.map_idx = map_idx

    def forward(self, x: np.ndarray):
        """ Run a single iteration of the SOM-algorithm.

        Args:
            x: The input features vector
        """
        # Calculate best matching unit for input x
        feat_diff = self.M - x[None, None, :]
        d_feat = np.linalg.norm(feat_diff, axis=-1)
        bmu_idx = np.array(np.unravel_index(np.argmin(d_feat), d_feat.shape))

        # Update neighbouring cells of BMU
        map_diff = self.map_idx - bmu_idx[None, None, :]
        d_map = np.linalg.norm(map_diff, axis=-1)
        h = self.alpha * np.exp(-0.5 * d_map / self.sigma ** 2)
        self.M += h[..., None] * (x - self.M)

    def adjust_hyperparams(self):
        """ Method to iteratively change the hyper-parameters of the SOM."""
        self.alpha += 0
        self.a += 0
        self.sigma = max(0.001, self.sigma - 0.1)

    def get_features(self, sample: Tuple[int, int], chips: np.ndarray, t: int):
        """ Get the feature vector a given sample of term-stimulus index pair.

        Args:
            sample: A pair of indices for sampled term and stimulus
            chips: The colour chip to LAB-space assignment
            t: Number of colour terms
        """
        t_idx, c_idx = sample
        term_feature = np.zeros(t)
        term_feature[t_idx] = self.a

        color_feature = chips[c_idx].flatten()

        return np.concatenate([term_feature, color_feature])

    def train(self, terms: pd.DataFrame, sems: np.ndarray, n: int):
        """ Train the SOM on the given data set and number of time steps.

        Args:
            terms: The term data
            sems: The semantic data
            n: The number of timesteps to train for
        """
        # Calculate the sampling distribution
        p_s_t = np.zeros((self.k, NUM_CHIPS))
        p_t = np.zeros(self.k)
        for i, (term, term_data) in enumerate(terms.groupby("word")):
            s_freq = term_data.groupby("chip").size().reindex(np.arange(1, 331), fill_value=0)
            p_s_t[i, :] = s_freq / term_data.shape[0]
            p_t[i] = term_data.shape[0]
        if self.p_t_estimator == "corpus":
            p_t /= p_t.sum()
        elif self.p_t_estimator == "uniform":
            p_t = 1 / self.k
        p_ts = p_s_t * p_t[:, None]

        # Calculate distance feature matrix
        d = get_distance_matrix(sems)

        index_matrix = np.arange(0, self.k * NUM_CHIPS)
        p_ts = p_ts.flatten()
        for i in np.arange(n):
            sample = tuple(np.unravel_index(np.random.choice(index_matrix, 1, p=p_ts), (self.k, NUM_CHIPS)))
            x = self.get_features(sample, d, self.k)
            self.forward(x)
            self.adjust_hyperparams()

    def predict(self, x: np.ndarray):
        """ Predict conditional term probabilities for each row in x. Only the final sem_size features are considered
        from each row.

         Args:
             x: Matrix of input features with each row being one sample.
         """
        x = get_distance_matrix(x)
        diff_x = self.M[:, :, None, -NUM_CHIPS:] - x[None, None, :]
        dist_x = np.linalg.norm(diff_x, axis=-1).reshape((-1, len(x)))
        bmu_idx = np.unravel_index(np.argmin(dist_x, axis=0), (self.m, self.m))
        bmu_x = self.M[bmu_idx]
        p_t_s = bmu_x[:, :self.k] / bmu_x[:, :self.k].sum(axis=-1, keepdims=True)
        return p_t_s


if __name__ == '__main__':
    # Global parameters
    save_xling = True  # Whether to save the cross-linguistic feature space after calculation

    # Define grid-search params here
    features = {"features": ["xling", "perc"],
                "sampling": ["corpus", "uniform"],
                "sigma": [5.0],
                "term_weight": [0.1, 0.3, 0.5],
                "alpha": [0.1, 0.3, 0.5],
                "size": {7, 10, 12}}

    glm = GaussianLanguageModel()
    for args in product_dict(**features):
        print(args)

        if args["features"] == "xling":
            sems = get_xling_features(glm.term_data, save=save_xling)
        elif args["features"] == "perc":
            sems = glm.chip_to_lab.to_numpy()
        else:
            raise RuntimeError("Invalid feature type given")

        for lang_id, terms in glm.term_data.groupby("language"):
            if lang_id != 32:
                continue

            t = terms.nunique()["word"]
            som = SelfOrganisingMap(term_size=t, **args)
            som.train(terms, sems, 20000)

            preds = som.predict(sems)
            mode_map(preds)
            plt.show()
