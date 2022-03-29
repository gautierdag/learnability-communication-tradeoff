import itertools
import os
import pickle
import argparse
import multiprocessing
from collections import defaultdict
from bidict import bidict
from typing import Tuple, List, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from noga.tools import MI, DKL

NUM_CHIPS = 330

# Number of samples to draw for each language
sample_range = (
        list(range(1, 25, 1))
        + list(range(25, 50, 5))
        + list(range(50, 100, 10))
        + list(range(100, 220, 20))
        + list(range(250, 1000, 50))
        + list(range(1000, 2100, 100))
        + list(range(3000, 100001, 1000))
)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class SelfOrganisingMap:
    """Define a self-organising map"""

    COLOR_PRIOR_EXCLUDED = [7, 19, 20, 25, 27, 31, 38, 48, 70, 80, 88, 91, 92, 93]

    def __init__(
            self,
            size: int = 12,
            alpha: float = 0.05,
            sigma: float = 5.0,
            term_weight: float = 0.1,
            wcs_path: str = "wcs",
            features: str = "perc",
            sampling: str = "corpus",
            color_prior: str = "capacity",
    ):
        """Initialise a new self-organising map.

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
        self.chip_to_lab: pd.DataFrame = self.chip_to_lab.sort_values(by="#cnum")[
            ["L*", "a*", "b*"]
        ]

        self.sample_size = 0
        self.term_data = pd.DataFrame()
        self.sem_data = pd.DataFrame()
        self.distance_matrix = np.empty(0)
        self.load_features(os.path.join(wcs_path, "term.txt"))

        self.term_size = {
            lid: data.nunique()["word"]
            for lid, data in self.term_data.groupby("language")
        }
        self.sem_size = self.sem_data.shape[1]
        self.models = {
            lid: np.zeros(
                (
                    self.size,
                    self.size,
                    self.term_size[lid] + self.distance_matrix.shape[0],
                )
            )
            for lid, _ in self.term_data.groupby("language")
        }

        # Frequentist sampling distribution
        self.word_map = defaultdict(bidict)
        self.pts = None
        self.get_sampling_distribution()

        # Get the same frequentist data distribution but as p(t|s)p(s)
        self.pt_s = None
        self.get_term_distribution()

        # Color prior (semantic space prior)
        self.ps = {}
        self.calculate_color_prior()
        ps = np.array([p for lid, p in self.ps.items()])
        ps = ps.sum(0) / len(ps)
        self.ps_universal = ps

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

    def reset_models(self):
        """ Reset every learnt model to all zeroes"""
        for mid, model in self.models.items():
            self.models[mid] = np.zeros_like(model, dtype=np.float64)

    def get_xling_features(self) -> np.ndarray:
        """Calculate the cross linguistic feature space for each colour chip."""
        if os.path.exists("xling.p"):
            return pickle.load(open("xling.p", "rb"))

        mat = []
        for lang_id, lang in self.term_data.groupby("language"):
            t = lang["word"].nunique()
            t_unique = lang["word"].dropna().unique()
            p_t_s = np.zeros((NUM_CHIPS, t))
            for i, (chip, chip_data) in enumerate(lang.groupby("chip")):
                c_freq = (
                    chip_data.groupby("word").size().reindex(t_unique, fill_value=0)
                )
                p_t_s[i, :] = c_freq / chip_data.shape[0]
            mat.append(p_t_s)
        mat = np.hstack(mat)

        if not os.path.exists("xling.p"):
            pickle.dump(mat, open("xling.p", "wb"))

        return mat

    def get_distance_matrix(self) -> np.ndarray:
        """Calculate the relative distance among the semantic features to use as input features."""
        data = self.sem_data.to_numpy() if self.features == "perc" else self.sem_data
        dist = np.zeros((NUM_CHIPS, NUM_CHIPS))
        for i, row in enumerate(data):
            for j, col in enumerate(data[i:], i):
                d = np.linalg.norm(row - col)
                dist[i, j] = d
        dist += dist.T
        return dist

    def get_sampling_distribution(self) -> Dict[int, np.ndarray]:
        """Calculate joint distribution of chips and terms from the data as p(s|t)p(t)"""
        # Calculate the sampling distribution
        dists = {}
        for lid, data in self.term_data.groupby("language"):
            size = self.term_size[lid]
            ps_t = np.zeros((size, NUM_CHIPS))
            pt = np.zeros(size)
            for i, (term, term_data) in enumerate(data.groupby("word")):
                s_freq = (
                    term_data.groupby("chip").size().reindex(np.arange(1, NUM_CHIPS + 1), fill_value=0)
                )
                ps_t[i, :] = s_freq / term_data.shape[0]
                pt[i] = term_data.shape[0]
                self.word_map[lid][i] = term
            if self.sampling == "corpus":
                pt /= pt.sum()
            elif self.sampling == "uniform":
                pt = 1 / size
            dists[lid] = ps_t * pt[:, None]
        self.pts = dists
        return self.pts

    def get_term_distribution(self):
        """Calculate the same term and chip distribution but as p(t|s)p(s)."""
        if os.path.exists("pt_s.p"):
            self.pt_s = pickle.load(open("pt_s.p", "rb"))
            return
        dists = {}
        for lid, data in self.term_data.groupby("language"):
            size = self.term_size[lid]
            data = data[~data.word.isna()]
            words = data["word"].unique()
            words.sort()
            pt_s = np.zeros((NUM_CHIPS, size))
            ps = np.zeros(NUM_CHIPS)
            for i, (chip, chip_data) in enumerate(data.groupby("chip")):
                t_freq = chip_data.groupby("word").size()
                t_freq = t_freq.reindex(words, fill_value=0)
                t_freq = t_freq[~t_freq.index.isna()]
                if np.allclose(t_freq, 0.0):
                    t_freq += 1  # Add one to each word then normalise to uniform distribution
                pt_s[i, :] = t_freq / t_freq.sum()
                ps[i] = chip_data.shape[0]
            if self.sampling == "corpus":
                ps /= ps.sum()
            elif self.sampling == "uniform":
                ps = 1 / size
            dists[lid] = pt_s  # * ps[:, None]
        self.pt_s = dists
        pickle.dump(dists, open("pt_s.p", "wb"))
        return self.pt_s

    def calculate_color_prior(self, maxiters=1000, verbose=False):
        """Calculate capacity-inducing prior over the colour chip space."""
        if self.color_prior == "uniform":
            self.ps = {lid: np.full(NUM_CHIPS, 1 / NUM_CHIPS)[:, None] for lid in self.pts}
        elif self.color_prior == "capacity":
            if os.path.exists("ps.p"):
                self.ps = pickle.load(open("ps.p", "rb"))
                return

            ps = {}
            for lid, p_y_x in tqdm(self.pt_s.items(), desc="Calculating capacity-achieving prior"):
                r_x = np.ones(p_y_x.shape[0]) / p_y_x.shape[0]
                r0 = np.zeros(p_y_x.shape[0])
                iters = 0
                while not np.all(np.isclose(r_x, r0)) and iters < maxiters:
                    iters += 1
                    r0 = r_x
                    q_xy = p_y_x * r_x[:, np.newaxis] / (p_y_x * r_x[:, np.newaxis]).sum()
                    q_x_y = q_xy / np.sum(q_xy, axis=0, keepdims=True)
                    r_x = np.prod(np.power(q_x_y, p_y_x), axis=1)
                    r_x = r_x / r_x.sum()
                    if verbose:
                        print(iters, np.round(((r_x - r0) ** 2).sum() ** 0.5, 10))
                ps[lid] = r_x
            self.ps = ps[:, None]
            pickle.dump(ps, open("ps.p", "wb"))

    def forward(self, m: np.ndarray, x: np.ndarray):
        """Run a single iteration of the SOM-algorithm.

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
        """Method to iteratively change the hyper-parameters of the SOM."""
        self.alpha += 0
        self.a += 0
        self.sigma = max(0.001, self.sigma - 0.01)

    def reset_hyperparams(self):
        """Reset hyper-parameters to their original values."""
        self.alpha = self._s_alpha
        self.a = self._s_a
        self.sigma = self._s_sigma

    def get_features(self, sample: Tuple[int, int], term_size: int):
        """Get the feature vector a given sample of term-stimulus index pair.

        Args:
            sample: A pair of indices for sampled term and stimulus
            term_size: The number of unique terms in the language
        """
        t_idx, c_idx = sample
        term_feature = np.zeros(term_size)
        term_feature[t_idx] = self.a

        color_feature = self.distance_matrix[c_idx].flatten()

        return np.concatenate([term_feature, color_feature])

    def get_wcs_form(
            self, samples: Tuple[np.ndarray, np.ndarray], language_id: int
    ) -> pd.DataFrame:
        """Return the samples in the WCS form."""
        wcs_samples = []
        for sample in zip(*samples):
            word_idx, chip_idx = sample

            speaker = 1
            color = chip_idx + 1
            word = self.word_map[language_id][word_idx]

            wcs_samples.append((language_id, speaker, color, word))
        return pd.DataFrame(
            wcs_samples, columns=["language", "speaker", "chip", "word"]
        )

    def score(self,
              pts: np.ndarray,
              model: np.ndarray,
              ps: np.ndarray,
              n_words: int,
              save=None) -> np.ndarray:
        """Score the current model for information loss and complexity

        Args:
            pts: The true data distribution
            model: The learnt SOM model
            ps: The prior over chips
            n_words: The number of colour terms
            save: If not None, then gives the file to save the calculated probability distribution to
        """
        pt_s_h = self.predict_t_s_model(self.distance_matrix, model, n_words)
        if save is not None:
            np.save(save, pt_s_h)
        pst_h = pt_s_h * ps

        inf_loss = DKL(pst_h.T, pts)
        mutual_info_h = MI(pst_h)

        return np.array([mutual_info_h, inf_loss])

    def learn_languages(
            self,
            n: int = 1000,
            scoring_steps: List[int] = None,
            language_ids: List[int] = None,
            save_samples: str = None,
            seed: int = 42,
            save_pt_s: str = None
    ) -> Dict[int, np.ndarray]:
        """Train the SOM on the given data set and number of time steps.

        Args:
            n: The number of time steps to train for
            scoring_steps: The time steps at which to score the models
            language_ids: The languages to train
            save_samples: The directory to save samples to
            seed: The seed of the simulation
            samples: If given, then learn using the given samples.

        Returns:
            A list of pairs of model scores with the same size as eval_steps
        """
        if language_ids is not None:
            scores = {lid: [] for lid in language_ids}
        else:
            scores = {lid: [] for lid in self.models}

        for lid, data in tqdm(
                self.term_data.groupby("language"), desc="Learning colours"
        ):
            if language_ids is not None and lid not in language_ids:
                continue
            size = self.term_size[lid]
            index_matrix = np.arange(0, size * NUM_CHIPS)
            pts = self.pts[lid]
            samples = tuple(
                np.unravel_index(
                    np.random.choice(index_matrix, n, p=pts.flatten()), (size, NUM_CHIPS)
                )
            )

            if save_samples is not None:
                path = os.path.join(os.path.join(save_samples, str(lid)))
                self.get_wcs_form(samples, lid).to_csv(
                    os.path.join(path, f"{n}_samples.csv"),
                    sep="\t",
                    index=False,
                    header=False,
                )

            m = self.models[lid]
            language_scores = self.learn_language_from_samples(
                lid, samples, scoring_steps, size, m, pts,
                os.path.join(save_pt_s, f"{lid}") if save_pt_s is not None else None)
            scores[lid] = language_scores
        return scores

    def learn_language_from_samples(self,
                                    language_id: int = None,
                                    samples: Tuple[List[int], List[int]] = None,
                                    scoring_steps: List[int] = None,
                                    n_words: int = None,
                                    m: np.ndarray = None,
                                    pts: np.ndarray = None,
                                    save_pt_s: str = None):
        """ Train the SOM on the given sample set. """
        scores = []
        if language_id is not None:
            ps_l = 0.9 * self.ps[language_id] + 0.1 * self.ps_universal
        else:
            ps_l = self.ps_universal
        for i, sample in tqdm(enumerate(zip(*samples), 1), desc=f"Language {language_id}"):
            x = self.get_features(sample, n_words)
            self.forward(m, x)
            self.adjust_hyperparams()

            if scoring_steps is not None and i in scoring_steps:
                scores.append(
                    self.score(
                        pts=pts,
                        model=m,
                        ps=ps_l,
                        n_words=n_words,
                        save=os.path.join(save_pt_s, f"{i}_pt_s.npy") if save_pt_s is not None else None,
                    )
                )
        self.reset_hyperparams()
        return np.array(scores)

    def predict_t_s(self, x: np.ndarray = None, language_ids: List[int] = None):
        """Predict conditional term probabilities for each row in x given a colour chip.
        Only the final sem_size features are considered from each row.

         Args:
             x: Data to predict on. If not given, run predictions for each colour chip.
             language_ids: The languages to run prediction for
        """
        if x is None:
            x = self.distance_matrix

        pt_s_arr = {}
        for lid, _ in self.term_data.groupby("language"):
            if language_ids is not None and lid not in language_ids:
                continue
            size = self.term_size[lid]
            m = self.models[lid]
            pt_s_arr[lid] = self.predict_t_s_model(x, m, size)
        return pt_s_arr

    def predict_t_s_model(self, x: np.ndarray, m: np.ndarray, n_words: int):
        diff_x = m[:, :, None, -NUM_CHIPS:] - x[None, None, :]
        dist_x = np.linalg.norm(diff_x, axis=-1).reshape((-1, len(x)))
        bmu_idx = np.unravel_index(
            np.argmin(dist_x, axis=0), (self.size, self.size)
        )
        bmu_x = m[bmu_idx]
        pt_s = bmu_x[:, :n_words] / bmu_x[:, :n_words].sum(axis=-1, keepdims=True)
        return pt_s

    def predict_s_t(self, x: np.ndarray = None, language_ids: List[int] = None):
        ps_t_arr = []
        for lid, _ in self.term_data.groupby("language"):
            if language_ids is not None and lid not in language_ids:
                continue

            size = self.term_size[lid]
            if x is None:
                x = np.eye(size) * self.a

            m = self.models[lid]
            ps_t_arr[lid] = self.predict_s_t_model(x, m, size)
        return ps_t_arr

    def predict_s_t_model(self, x: np.ndarray, m: np.ndarray, n_words: int):
        diff_x = m[:, :, None, :n_words] - x[None, None, :]
        dist_x = np.linalg.norm(diff_x, axis=-1).reshape((-1, len(x)))
        bmu_idx = np.unravel_index(
            np.argmin(dist_x, axis=0), (self.size, self.size)
        )
        bmu_x = m[bmu_idx]
        ps_t = bmu_x[:, n_words:] / bmu_x[:, n_words:].sum(axis=-1, keepdims=True)
        return ps_t


def get_average_scores(scores: List[Dict[int, np.ndarray]]) -> Dict[int, np.ndarray]:
    dic = defaultdict(list)
    for s_dict in scores:
        for lid, arr in s_dict.items():
            dic[lid].append(arr)

    ret = {}
    for lid, arr in dic.items():
        np_arr = np.array(arr)
        ret[lid] = np.hstack([np.mean(np_arr, 0), np.std(np_arr, 0)])
    return ret


def func(args):
    som_ = SelfOrganisingMap(**args[0])
    seed_ = args[1]
    sample_range_ = args[2]
    lids_ = args[3]
    scores_ = som_.learn_languages(
        sample_range_[-1],
        scoring_steps=sample_range_,
        language_ids=lids_,
        save_samples=None,
        save_pt_s=None,
        seed=seed_,
    )
    return scores_, som_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOM on WCS data")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--average_k", type=int, default=5, help="The number of learners to "
                                                                 "average over for the developmental plots.")
    parser.add_argument("--workers", type=int, default=None, help="If given, then use multiprocessing with "
                                                                  "given number of workers.")
    parser.add_argument("--lid", type=int, default=None, help="ID of language to learn.")

    args = parser.parse_args()

    # Global parameters
    seed = args.seed
    save_xling = True  # Whether to save the cross-linguistic feature space
    grid_search = False
    save_p = True
    save_samples = False

    # lids = list(range(1, 110))
    if args.lid is not None:
        lids = [args.lid]
    else:
        lids = [2, 32, 35, 108]
    # lids = [2]

    np.seterr(divide="ignore")
    np.random.seed(seed)

    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists("output/som"):
        os.mkdir("output/som")
    if not os.path.exists(f"output/som/{seed}"):
        os.mkdir(f"output/som/{seed}")
    for lid in lids:
        if not os.path.exists(f"output/som/{seed}/{lid}"):
            os.mkdir(f"output/som/{seed}/{lid}")

    optimal_hyper_params = pickle.load(open("grid_search_params.p", "rb"))

    for lid in lids:
        som_args = optimal_hyper_params[lid]
        print(som_args)

        scores = []
        models = []

        if args.workers is not None:
            with multiprocessing.Pool(processes=args.workers) as p:
                scores_models = p.map(func, [(som_args, seed, sample_range, [lid])
                                             for i in range(args.average_k)])
            scores, models = list(zip(*scores_models))
        else:
            for k in trange(args.average_k):
                som = SelfOrganisingMap(**som_args)
                scores_dict = som.learn_languages(
                    sample_range[-1],
                    scoring_steps=sample_range,
                    language_ids=[lid],
                    save_samples=os.path.join("output", "som", str(seed)) if save_samples else None,
                    save_pt_s=f"output/som/{seed}/" if save_p else None,
                    seed=seed,
                )
                models.append(som)
                scores.append(scores_dict)

                # Load all saved p_t_s and join to already calculated ones
                if save_p:
                    for s in sample_range:
                        p_t_s = np.load(f"output/som/{seed}/{lid}/{s}_pt_s.npy")
                        if not os.path.exists(f"output/som/{seed}/{lid}/{s}_pt_s_all.npy"):
                            joined = p_t_s[None, :]
                        else:
                            joined = np.load(f"output/som/{seed}/{lid}/{s}_pt_s_all.npy")
                            joined = np.vstack([joined, p_t_s[None, :]])
                        np.save(f"output/som/{seed}/{lid}/{s}_pt_s_all.npy", joined)

        scores_dict = get_average_scores(scores)
        pickle.dump(scores_dict, open(f"output/som/{seed}/scores_dict.p", "wb"))
