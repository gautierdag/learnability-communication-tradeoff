from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learnability import GaussianLanguageModel
from noga.figures import mode_map


def get_xling_features(data):


class SelfOrganisingMap:
    """ Define a self-organising map"""
    def __init__(self, size: int, term_size: int, alpha: float, sigma: float, term_weight: float,
                 sem_size: int = 3, p_t_estimator: str = "rel_freq"):
        """ Initialise a new self-organising map.

        Args:
            size: The size of the square matrix M
            term_size: The number of term features
            sem_size: The number of semantic features
            alpha: The learning rate
            sigma: Neighbourhood radius
            term_weight: Term importance weight
            p_t_estimator: Estimation method for p(t). Either unif or rel_freq
        """
        assert size > 0, "Matrix size must be larger than 0."
        assert term_size > 0, "Features size must be larger than 0."
        assert 0 < alpha < 1, "Learning rate alpha must be between 0 and 1."
        assert 0 < sigma, "Neighbourhood radius must be positive."
        assert 0 < term_weight <= 1, "Term importance must be in (0,1]."

        self.m = size
        self.k = term_size
        self.d = sem_size
        self.M = np.zeros((self.m, self.m, self.k + 3))  # Add 3 for LAB space
        self.alpha = alpha
        self.sigma = sigma
        self.a = term_weight
        self.p_t_estimator = p_t_estimator

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

    def get_features(self, sample: Tuple[int, int], chips: pd.DataFrame, t: int):
        """ Get the feature vector a given sample of term-stimulus index pair.

        Args:
            sample: A pair of indices for sampled term and stimulus
            chips: The colour chip to LAB-space assignment
            t: Number of colour terms
        """
        t_idx, c_idx = sample
        term_feature = np.zeros(t)
        term_feature[t_idx] = self.a
        color_feature = chips.iloc[c_idx].to_numpy().flatten()
        return np.concatenate([term_feature, color_feature])

    def train(self, data: pd.DataFrame, chips: pd.DataFrame, n: int):
        """ Train the SOM on the given data set and number of time steps.

        Args:
            data: The data to sample
            chips: The colour chip to LAB-space assignment
            n: The number of timesteps to train for
        """
        # Calculate the sampling distribution
        C = 330

        p_s_t = np.zeros((self.k, C))
        p_t = np.zeros(self.k)
        for i, (term, term_data) in enumerate(data.groupby("word")):
            s_freq = term_data.groupby("chip").size().reindex(np.arange(1, 331), fill_value=0)
            p_s_t[i, :] = s_freq / term_data.shape[0]
            p_t[i] = term_data.shape[0]
        if self.p_t_estimator == "rel_freq":
            p_t /= p_t.sum()
        elif self.p_t_estimator == "unif":
            p_t = 1 / self.k
        p_ts = p_s_t * p_t[:, None]

        index_matrix = np.arange(0, self.k * C)
        p_ts = p_ts.flatten()
        for i in np.arange(n):
            sample = tuple(np.unravel_index(np.random.choice(index_matrix, 1, p=p_ts), (self.k, C)))
            x = self.get_features(sample, chips, self.k)
            self.forward(x)
            self.adjust_hyperparams()

    def predict(self, x: np.ndarray):
        """ Predict conditional term probabilities for each row in x. Only the final sem_size features are considered
        from each row.

         Args:
             x: Matrix of input features with each row being one sample.
         """
        x = x[:, -self.d:]
        diff_x = self.M[:, :, None, -self.d:] - x[None, None, :]
        dist_x = np.linalg.norm(diff_x, axis=-1).reshape((-1, len(x)))
        bmu_idx = np.unravel_index(np.argmin(dist_x, axis=0), (self.m, self.m))
        bmu_x = self.M[bmu_idx]
        p_t_s = bmu_x[:, :self.k] / bmu_x[:, :self.k].sum(axis=-1, keepdims=True)
        return p_t_s


if __name__ == '__main__':
    glm = GaussianLanguageModel()
    for lang_id, data in glm.term_data.groupby("language"):
        if lang_id != 108:
            continue

        t = data.nunique()["word"]
        som = SelfOrganisingMap(12, t, alpha=0.1, sigma=5, term_weight=0.3)
        som.train(data, glm.chip_to_lab, 10000)
        p_t_s = som.predict(glm.chip_to_lab.to_numpy())
        mode_map(p_t_s)
        plt.show()
