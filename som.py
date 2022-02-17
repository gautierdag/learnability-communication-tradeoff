import itertools
import os
import pickle
from collections import defaultdict
from typing import Tuple, List, Dict, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from ck_blahut_arimoto import ck_blahut_arimoto_ib
from learnability import GaussianLanguageModel
from noga.figures import mode_map
from noga.tools import DKL, MI


NUM_CHIPS = 330
COLOR_PRIOR_EXCLUDED = [7, 19, 20, 25, 27, 31, 38, 48, 70, 80, 88, 91, 92, 93]


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class SelfOrganisingMap:
    """ Define a self-organising map"""

    COLOR_PRIOR_EXCLUDED = [7, 19, 20, 25, 27, 31, 38, 48, 70, 80, 88, 91, 92, 93]

    def __init__(self,
                 size: int,
                 alpha: float,
                 sigma: float,
                 term_weight: float,
                 wcs_path: str = "wcs",
                 features: str = "perc",
                 sampling: str = "corpus",
                 color_prior: str = "uniform"):
        """ Initialise a new self-organising map.

        Args:
            wcs_path: Path to the directory containing the WCS data
            size: The size of the square matrix M
            alpha: The learning rate
            sigma: Neighbourhood radius
            term_weight: Term importance weight
            features: Type of semantic features to use. Either 'perc' or 'xling'
            sampling: Estimation method for p(t). Either unif or rel_freq
            color_prior: Optional semantic space (colour) prior
        """
        assert size > 0, "Matrix size must be larger than 0."
        assert 0 < alpha < 1, "Learning rate alpha must be between 0 and 1."
        assert 0 < sigma, "Neighbourhood radius must be positive."
        assert 0 < term_weight <= 1, "Term importance must be in (0,1]."
        assert features in ["perc", "xling"], "Invalid feature type given."
        assert sampling in ["corpus", "uniform"], "Invalid estimation type for p(t)."
        assert color_prior in ["uniform", "capacity"], f"Invalid color prior specified."

        self.size = size
        self.alpha = alpha
        self._s_alpha = alpha
        self.sigma = sigma
        self._s_sigma = sigma
        self.a = term_weight
        self._s_a = term_weight
        self.features = features
        self.sampling = sampling
        self.color_prior = color_prior

        # Load the data
        self.wcs_path = wcs_path
        self.chip_data = pd.read_csv(
            os.path.join(wcs_path, "chip.txt"),
            sep="\t",
            index_col=0,
            names=["row", "col", "pos"],
        )
        self.chip_to_lab = pd.read_csv(
            os.path.join(wcs_path, "cnum-vhcm-lab-new.txt"),
            sep="\t",
            header=0,
            index_col="#cnum",
        )
        self.chip_to_lab: pd.DataFrame = self.chip_to_lab.sort_values(by="#cnum")[["L*", "a*", "b*"]]

        self.sample_size = 0
        self.term_data = pd.DataFrame()
        self.sem_data = pd.DataFrame()
        self.distance_matrix = np.empty(0)
        self.load_features(os.path.join(wcs_path, "term.txt"))

        self.term_size = {lid: data.nunique()["word"] for lid, data in self.term_data.groupby("language")}
        self.sem_size = self.sem_data.shape[1]
        self.models = {lid: np.zeros((self.size, self.size, self.term_size[lid] + self.distance_matrix.shape[0]))
                       for lid, _ in self.term_data.groupby("language")}

        # Frequentist sampling distribution
        self.pst = None
        self.get_sampling_distribution()

        # Color prior (semantic space prior)
        self.ps = None
        self.calculate_color_prior()

        # Get the same frequentist data distribution but as p(t|s)p(s)
        self.pts = None
        # self.get_term_distribution()

        # Map coordinate array for calculating d_map
        map_idx = np.arange(0, self.size ** 2).reshape((self.size, self.size))
        map_idx = np.stack(np.unravel_index(map_idx, (self.size, self.size)), axis=2)
        self.map_idx = map_idx

    def load_features(self, term_file: Union[str, pd.DataFrame]):
        """Load new color term data to the model, overwriting the current data.

        Args:
            term_file: Either a string giving the path to the term data or a DataFrame containing already read data
        """
        if isinstance(term_file, str):
            self.term_data = pd.read_csv(
                term_file, sep="\t", names=["language", "speaker", "chip", "word"]
            )
        elif isinstance(term_file, pd.DataFrame):
            self.term_data = term_file
        else:
            return

        self.sample_size = dict(self.term_data.groupby("language").size())

        if self.features == "xling":
            self.sem_data = self.get_xling_features()
        elif self.features == "perc":
            self.sem_data = self.chip_to_lab

        self.distance_matrix = self.get_distance_matrix()

    def get_xling_features(self) -> np.ndarray:
        """ Calculate the cross linguistic feature space for each colour chip. """
        if os.path.exists("xling.p"):
            return pickle.load(open("xling.p", "rb"))

        mat = []
        for lang_id, lang in self.term_data.groupby("language"):
            t = lang["word"].nunique()
            t_unique = lang["word"].dropna().unique()
            p_t_s = np.zeros((NUM_CHIPS, t))
            for i, (chip, chip_data) in enumerate(lang.groupby("chip")):
                c_freq = chip_data.groupby("word").size().reindex(t_unique, fill_value=0)
                p_t_s[i, :] = c_freq / chip_data.shape[0]
            mat.append(p_t_s)
        mat = np.hstack(mat)

        if not os.path.exists("xling.p"):
            pickle.dump(mat, open("xling.p", "wb"))

        return mat

    def get_distance_matrix(self) -> np.ndarray:
        """ Calculate the relative distance among the semantic features to use as input features. """
        data = self.sem_data.to_numpy()
        dist = np.zeros((NUM_CHIPS, NUM_CHIPS))
        for i, row in enumerate(data):
            for j, col in enumerate(data[i:], i):
                d = np.linalg.norm(row - col)
                dist[i, j] = d
        dist += dist.T
        return dist

    def get_sampling_distribution(self) -> Dict[int, np.ndarray]:
        """ Calculate joint distribution of chips and terms from the data as p(s|t)p(t) """
        # Calculate the sampling distribution
        dists = {}
        for lid, data in self.term_data.groupby("language"):
            size = self.term_size[lid]
            ps_t = np.zeros((size, NUM_CHIPS))
            pt = np.zeros(size)
            for i, (term, term_data) in enumerate(data.groupby("word")):
                s_freq = term_data.groupby("chip").size().reindex(np.arange(1, NUM_CHIPS + 1), fill_value=0)
                ps_t[i, :] = s_freq / term_data.shape[0]
                pt[i] = term_data.shape[0]
            if self.sampling == "corpus":
                pt /= pt.sum()
            elif self.sampling == "uniform":
                pt = 1 / size
            dists[lid] = ps_t * pt[:, None]
        self.pst = dists
        return self.pst

    def get_term_distribution(self):
        """ Calculate the same term and chip distribution but as p(t|s)p(s)."""
        dists = {}
        for lid, data in self.term_data.groupby("language"):
            size = self.term_size[lid]
            words = data["word"].unique()
            pt_s = np.zeros((NUM_CHIPS, size))
            ps = np.zeros(NUM_CHIPS)
            for i, (chip, chip_data) in enumerate(data.groupby("chip")):
                t_freq = chip_data.groupby("word").size()
                t_freq = t_freq.reindex(words, fill_value=0)
                t_freq = t_freq[~t_freq.index.isna()]
                pt_s[i, :] = t_freq / t_freq.sum()
                ps[i] = chip_data.shape[0]
            if self.sampling == "corpus":
                ps /= ps.sum()
            elif self.sampling == "uniform":
                ps = 1 / size
            dists[lid] = pt_s * ps[:, None]
        self.pts = dists
        return self.pts

    def calculate_color_prior(self):
        """Calculate capacity-inducing prior over the colour chip space."""
        if self.color_prior == "uniform":
            self.ps = np.full(NUM_CHIPS, 1 / NUM_CHIPS)
        elif self.color_prior == "capacity":
            if os.path.exists("ps.p"):
                self.ps = pickle.load(open("ps.p", "rb"))[:, None]
                return

            L = 0
            ps = np.zeros(self.chip_data.shape[0])
            for lid, pst in tqdm(self.pst.items(), desc="Calculating capacity-inducing prior"):
                if lid in self.COLOR_PRIOR_EXCLUDED:
                    continue
                _, q_wst = ck_blahut_arimoto_ib(
                    pst, 1, np.full(pst.shape, 1 / pst.size), "kl-divergence"
                )
                pls = q_wst.sum(axis=0).sum(axis=1)
                ps += pls
                L += 1
            ps /= L
            self.ps = ps[:, None]
            pickle.dump(ps, open("ps.p", "wb"))

    def forward(self, m: np.ndarray, x: np.ndarray):
        """ Run a single iteration of the SOM-algorithm.

        Args:
            m: The matrix to update
            x: The input features vector
        """
        # Calculate best matching unit for input x
        feat_diff = m - x[None, None, :]
        d_feat = np.linalg.norm(feat_diff, axis=-1)
        bmu_idx = np.array(np.unravel_index(np.argmin(d_feat), d_feat.shape))

        # Update neighbouring cells of BMU
        map_diff = self.map_idx - bmu_idx[None, None, :]
        d_map = np.linalg.norm(map_diff, axis=-1)
        h = self.alpha * np.exp(-0.5 * d_map / self.sigma ** 2)
        m += h[..., None] * (x - m)

    def adjust_hyperparams(self):
        """ Method to iteratively change the hyper-parameters of the SOM."""
        self.alpha += 0
        self.a += 0
        self.sigma = max(0.001, self.sigma - 0.1)

    def reset_hyperparams(self):
        """ Reset hyper-parameters to their original values. """
        self.alpha = self._s_alpha
        self.a = self._s_a
        self.sigma = self._s_sigma

    def get_features(self, sample: Tuple[int, int], term_size: int):
        """ Get the feature vector a given sample of term-stimulus index pair.

        Args:
            sample: A pair of indices for sampled term and stimulus
            term_size: The number of unique terms in the language
        """
        t_idx, c_idx = sample
        term_feature = np.zeros(term_size)
        term_feature[t_idx] = self.a

        color_feature = self.distance_matrix[c_idx].flatten()

        return np.concatenate([term_feature, color_feature])

    def get_wcs_form(self, samples: np.ndarray) -> pd.DataFrame:
        """ Return the samples in the WCS form."""
        words_list = list(self.models_params[lang_id].keys())
        for idx in sample_idx:
            word_idx, chip_idx = idx // pwc.shape[1], idx % pwc.shape[1]

            speaker = 1
            color = chip_idx + 1
            word = words_list[word_idx]

            samples.append((lang_id, speaker, color, word))

    def score(self, samples: np.ndarray, language_id: int) -> np.ndarray:
        """ Score the current model for information loss and complexity """
        pts_h = self.predict(language_ids=[language_id])[0] * self.ps

        # Compute information loss as KL-divergence from adult model
        inf_loss = DKL(self.pst[language_id], pts_h.T)
        # likelihood = self.predict(samples[:, self.term_size[language_id]:], [language_id])[0]
        # ll = np.log(likelihood / likelihood.sum())
        # ll[ll == -np.inf] = 0
        # inf_loss = np.sum(ll)

        # Compute complexity (mutual information)
        ph = 1
        mutual_info_h = MI(pts_h * ph)

        return np.array([mutual_info_h, inf_loss])

    def learn_languages(self, n: int, scoring_steps: List[int] = None, language_ids: List[int] = None) \
            -> Dict[int, List[Tuple[float, float]]]:
        """ Train the SOM on the given data set and number of time steps.

        Args:
            n: The number of time steps to train for
            scoring_steps: The time steps at which to score the models
            language_ids: The languages to train

        Returns:
            A list of pairs of model scores with the same size as eval_steps
        """
        scores = defaultdict(list)
        for lid, data in tqdm(self.term_data.groupby("language"), desc="Learning colours"):
            if language_ids is not None and lid not in language_ids:
                continue
            size = self.term_size[lid]
            index_matrix = np.arange(0, size * NUM_CHIPS)
            pts = self.pst[lid].flatten()
            samples = tuple(np.unravel_index(np.random.choice(index_matrix, n, p=pts), (size, NUM_CHIPS)))

            samples_seen = []
            m = self.models[lid]
            for i, sample in tqdm(enumerate(zip(*samples)), desc=f"Language {lid}"):
                x = self.get_features(sample, size)
                samples_seen.append(x)
                self.forward(m, x)
                self.adjust_hyperparams()

                if scoring_steps is not None and i in scoring_steps:
                    scores[lid].append(self.score(np.array(samples_seen), language_id=lid))
            self.reset_hyperparams()
        return scores

    def predict(self, x: np.ndarray = None, language_ids: List[int] = None):
        """ Predict conditional term probabilities for each row in x given a colour chip.
        Only the final sem_size features are considered from each row.

         Args:
             x: Data to predict on. If not given, run predictions for each colour chip.
             language_ids: The languages to run prediction for
         """
        if x is None:
            x = self.distance_matrix

        pt_s_arr = []
        for lid, _ in self.term_data.groupby("language"):
            if language_ids is not None and lid not in language_ids:
                continue
            size = self.term_size[lid]
            m = self.models[lid]
            diff_x = m[:, :, None, -NUM_CHIPS:] - x[None, None, :]
            dist_x = np.linalg.norm(diff_x, axis=-1).reshape((-1, len(x)))
            bmu_idx = np.unravel_index(np.argmin(dist_x, axis=0), (self.size, self.size))
            bmu_x = m[bmu_idx]
            pt_s = bmu_x[:, :size] / bmu_x[:, :size].sum(axis=-1, keepdims=True)
            pt_s_arr.append(pt_s)
        return pt_s_arr


if __name__ == '__main__':
    # Global parameters
    seed = 42
    save_xling = True  # Whether to save the cross-linguistic feature space after calculation
    grid_search = False
    save_samples = False

    np.seterr(divide="ignore")
    np.random.seed(seed)

    if grid_search:
        features = {"features": ["xling", "perc"],
                    "sampling": ["corpus", "uniform"],
                    "sigma": [5.0],
                    "term_weight": [0.1, 0.3, 0.5],
                    "alpha": [0.1, 0.3, 0.5],
                    "size": {7, 10, 12},
                    "color_prior": ["capacity", "uniform"]}
    else:
        features = {"features": ["perc"],
                    "sampling": ["corpus"],
                    "sigma": [5.0],
                    "term_weight": [0.3],
                    "alpha": [0.1],
                    "size": [12],
                    "color_prior": ["capacity"]}

    # Number of samples to draw for each language
    sample_range = (
        list(range(1, 25, 1))
        + list(range(25, 50, 5))
        + list(range(50, 100, 10))
        + list(range(100, 220, 20))
        + list(range(250, 1000, 50))
        + list(range(1000, 2100, 100))
        # + list(range(3000, 10001, 1000))
        # + list(range(20000, 100001, 10000))
    )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    lids = [2]

    for args in product_dict(**features):
        print(args)
        som = SelfOrganisingMap(**args)
        scores_dict = som.learn_languages(sample_range[-1], scoring_steps=sample_range, language_ids=lids)

        plt.figure()
        preds = som.predict(language_ids=lids)
        for pred in preds:
            mode_map(pred)
            plt.show()

        for lid, scores in scores_dict.items():
            scores = np.array(scores)
            plt.quiver(*scores[:-1].T, *np.diff(scores, axis=0).T,
                      angles='xy', scale_units='xy', scale=1,
                      width=0.005, headwidth=2, color=colors[lid % len(colors)]
                      )
            plt.scatter(scores[:, 0], scores[:, 1], s=6,
                       edgecolor="white", linewidth=0.5)
            plt.xlabel("Complexity; $I(H, C)$ bits")
            plt.ylabel("Information Loss; Log likelihood")
            plt.gcf().tight_layout()

            plt.show()
