from collections import defaultdict
import pickle

import imageio
import os
from typing import List, Dict, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from ck_blahut_arimoto import ck_blahut_arimoto_ib
from noga.figures import grid2img, mode_map
from noga.tools import DKL, MI
from tqdm import tqdm


def plot_color_prior(pc: np.array) -> None:
    """Plot the color matrix in the WCS format.

    Args:
        pc: The prior distribution over colors.
    """
    grid = np.repeat(pc[:, None], 3, axis=1) / pc.max()
    img = np.flipud(grid2img(grid))
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    clrs = np.array([r.flatten(), g.flatten(), b.flatten()]).T
    ax = plt.pcolor(r, color=clrs, linewidth=0.04, edgecolors="None")
    ax.set_array(None)
    plt.xlim([0, 42])
    plt.ylim([0, 10])
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.show()


class GaussianLanguageModel:
    """Load and fit Gaussian models to each language from data stored in the WCS format."""

    # Excluded language IDs for color prior calculation, as given by Noga.
    COLOR_PRIOR_EXCLUDED = [7, 19, 20, 25, 27, 31, 38, 48, 70, 80, 88, 91, 92, 93]

    def __init__(
        self,
        term_file: Union[str, pd.DataFrame] = "wcs/term.txt",
        wcs_path: str = "wcs",
        cov_reg: float = 1.0e-5,
        cov_prior: np.ndarray = None,
    ):
        """Initialise a new model class. Language models are stored in the dictionary self.models which is indexed by
        the ID of the language.

        Args:
            term_file: The file containing the color term elicitations or a DataFrame with already read data
            wcs_path: Path to the directory containing the WCS data
            cov_reg: Covariance regularisation hyper-parameter
            cov_prior: Pre-calculated covariance matrix, used if calculated covariance matrix is degenerate
        """
        self.wcs_path = wcs_path
        self.cov_reg = cov_reg
        self.cov_prior = cov_prior
        self.models = {}
        self.models_unnormed = {}
        self.models_params = {}

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
        self.chip_to_lab = self.chip_to_lab.sort_values(by="#cnum")[["L*", "a*", "b*"]]

        self.sample_size = None
        self.term_data = None
        self.load_term_data(term_file)

    def load_term_data(self, term_file: Union[str, pd.DataFrame]):
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
        self.term_data = self.term_data.merge(
            self.chip_to_lab, left_on="chip", right_on="#cnum", how="left"
        )

    def calculate_color_prior(self) -> np.array:
        """Calculate capacity-inducing prior over the colour chip space."""
        assert self.models != {}, "No languages learnt. Try calling learn_lanuages()."

        L = 0
        pc = np.zeros(self.chip_data.shape[0])

        for lang_id, pwc in self.models.items():
            if lang_id in self.COLOR_PRIOR_EXCLUDED:
                continue

            _, q_wct = ck_blahut_arimoto_ib(
                pwc.to_numpy(), 1, np.full(pwc.shape, 1 / pwc.size), "kl-divergence"
            )
            plc = q_wct.sum(axis=0).sum(axis=1)
            pc += plc
            L += 1

        pc /= L
        return pc

    def calculate_cov_prior(self) -> np.ndarray:
        """Calculate covariance prior averaged over all languages and color terms."""
        prior = np.zeros((3, 3))
        for model in self.models_params.values():
            prior += sum([params[1] for params in model.values()]) / len(model)
        return prior / len(self.models_params)

    def learn_languages(
        self, language_ids: List[int] = None, progress_bar: bool = True
    ) -> None:
        """Fit a Gaussian model to each language to calculate the joint distribution P(W, C). Fitted models are stored
        in the field self.models

        Args:
            language_ids: Only fit to the languages with the given IDs
            progress_bar: Show progress bar
        """
        if progress_bar:
            progress = tqdm(self.term_data.groupby("language"))
        else:
            progress = self.term_data.groupby("language")

        for i, (lang_id, data) in enumerate(progress):
            if language_ids is not None and lang_id not in language_ids:
                continue

            nd = data.shape[0]  # Number of datapoints for this language
            nw = data["word"].nunique()  # Number of unique colour terms

            if progress_bar:
                progress.set_postfix({"lang": lang_id, "nd": nd, "nw": nw})

            # Fit Gaussian models to the color chips to get P(C|W)
            pc_w = []
            model_params = {}
            for ct, subset in data.groupby("word"):
                subset = subset[["L*", "a*", "b*"]]

                mu, cov = subset.mean().to_numpy(), subset.cov().to_numpy()
                if subset.shape[0] == 1:
                    if self.cov_prior is not None:
                        cov = self.cov_prior
                    else:
                        cov = np.identity(3)
                cov += np.identity(3) * self.cov_reg
                proba = multivariate_normal(mu, cov).pdf(self.chip_to_lab)

                model_params[ct] = (mu, cov)
                pc_w.append(proba / sum(proba))
            pc_w = np.array(pc_w)
            pwc = pc_w / pc_w.sum()

            self.models_unnormed[lang_id] = pd.DataFrame(
                pc_w, index=model_params.keys()
            )
            self.models[lang_id] = pd.DataFrame(pwc, index=model_params.keys())
            self.models_params[lang_id] = model_params

    def sample_languages(self, n: int = 5) -> pd.DataFrame:
        """Sample languages from the learned models.

        Args:
            n: The number of samples to draw from each languages' joint distribution.

        Returns: A pandas dataframe in the format of the wcs/terms.txt file.
        """
        samples = []

        for lang_id, pwc in self.models.items():
            sample_idx = np.random.choice(
                np.arange(pwc.size), (n,), True, pwc.to_numpy().flatten()
            )
            words_list = list(self.models_params[lang_id].keys())
            for idx in sample_idx:
                word_idx, chip_idx = idx // pwc.shape[1], idx % pwc.shape[1]

                speaker = 1
                color = chip_idx + 1
                word = words_list[word_idx]

                samples.append((lang_id, speaker, color, word))

        return pd.DataFrame(samples, columns=["language", "speaker", "chip", "word"])

    @staticmethod
    def write_samples_file(
        samples: pd.DataFrame, output_file: str = "sampled_term.txt"
    ) -> None:
        """Write a language sample to file in the wcs/term.txt format."""
        samples.to_csv(output_file, sep="\t", index=False, header=False)

    @staticmethod
    def score_languages(
        adult: "GaussianLanguageModel",
        child: "GaussianLanguageModel",
        color_prior: np.array = None,
    ) -> Dict[int, np.ndarray]:
        """For each real-world language score a hypothetical child language learnt on restricted number of sampled data
        against an adult language learnt on complete data.

        Args:
             adult: The adult model learnt on complete data
             child: The child model learnt on sampled, partial data
             color_prior: The color prior

        Returns:
            A dictionary of tuples in the form (mutual information, log-likelihood) for each language id.
        """
        scores = {}
        # samples = child.term_data.groupby("language")
        for lang_id in child.models:
            pwc_h = child.models[lang_id].to_numpy()
            pwc = adult.models[lang_id].to_numpy()

            # Simplicity prior
            n = child.sample_size[lang_id]
            nw = len(child.models_params[lang_id])
            ph = 1 / np.sqrt(nw)
            ph = simplicity_prior(*zip(*child.models_params[lang_id].values()))

            pwc_h_ = np.zeros_like(pwc)
            shared_idx = [
                i
                for i, w in enumerate(adult.models_params[lang_id])
                if w in child.models_params[lang_id]
            ]
            pwc_h_[shared_idx, :] = pwc_h
            inf_loss = DKL(pwc, pwc_h_)

            # Compute complexity (mutual information)
            pc_h = pwc_h.sum(axis=0)
            pc = pwc.sum(axis=0) if color_prior is None else color_prior
            mutual_info_h = np.nansum(pc_h * ph * (np.log(pc_h) - np.log(pc)))

            scores[lang_id] = np.array([mutual_info_h, inf_loss])
        return scores

 def simplicity_prior(mus, covs):
     tot_bc = 0
     for i in range(len(mus)-1):
         for j in range(i+1, len(mus)):
             comb_cov = (covs[i] + covs[j])/2
             bd = (1/8)*(mus[i] - mus[j]).T @ np.linalg.inv(comb_cov) @ (mus[i] - mus[j]) + (1/2) * np.log( np.linalg.det(comb_cov) / (np.sqrt(np.linalg.det(covs[i]) + np.linalg.det(covs[j]))) )
             bc = np.exp(-bd)
             tot_bc += bc
     return expon.pdf(len(mus) + tot_bc, scale=10)


if __name__ == "__main__":
    seed = 42
    np.seterr(divide="ignore")
    np.random.seed(seed)

    if not os.path.exists(f"output/learnability/{seed}/"):
        os.makedirs(f"output/learnability/{seed}/")

    adult_model = GaussianLanguageModel()
    adult_model.learn_languages()
    adult_cov_prior = adult_model.calculate_cov_prior()

    # Calculate color prior once
    if not os.path.exists("pc.p"):
        pc = adult_model.calculate_color_prior()
        plot_color_prior(pc)
        pickle.dump(pc, open("pc.p", "wb"))
    else:
        pc = pickle.load(open("pc.p", "rb"))

    # Run scoring function on various language samples
    scores = defaultdict(list)
    n_range = np.arange(1, 101, 1)
    language_ids = list(range(1, 111, 1))  # run over all languages
    lid = 47  # The language to examine

    child_model = GaussianLanguageModel(cov_prior=adult_cov_prior)
    samples = pd.DataFrame()

    plot_color_map = False
    color_maps = []

    for i in tqdm(n_range):
        sample = adult_model.sample_languages(1)
        samples = pd.concat([samples, sample], axis=0)
        samples = samples.sort_values("language")
        # GaussianLanguageModel.write_samples_file(sample)
        # child_model = GaussianLanguageModel("sampled_term.txt")
        child_model.load_term_data(samples)
        child_model.learn_languages(language_ids=language_ids, progress_bar=False)

        for lid in language_ids:
            if not os.path.exists(f"output/learnability/{seed}/{lid}/"):
                os.makedirs(f"output/learnability/{seed}/{lid}")

            s = GaussianLanguageModel.score_languages(adult_model, child_model, pc)[lid]
            scores[lid].append(s)
            with open(f"output/learnability/{seed}/{lid}/{i}.npy", "wb") as f:
                np.save(f, child_model.models_unnormed[lid])

        if plot_color_map:
            mode_map(
                child_model.models[lid].T.to_numpy(),
                adult_model.models[lid].sum(axis=0)[:, None],
            )
            fig = plt.gcf()
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            figure_size = tuple(
                np.array(fig.get_size_inches()[::-1] * fig.dpi, dtype=int)
            ) + (3,)
            image_from_plot = image_from_plot.reshape(figure_size)
            color_maps.append(image_from_plot)

    for lid in language_ids:
        scores_array = np.array(scores[lid])
        with open(f"output/learnability/{seed}/{lid}/scores.npy", "wb") as f:
            np.save(f, scores_array)

    if plot_color_map:
        with imageio.get_writer(f"lang_{lid}.gif", mode="I") as writer:
            for img in color_maps:
                writer.append_data(img)

    plt.scatter(scores_array[:, 0], scores_array[:, 1], c=n_range, cmap="gray_r")
    for i, coord in enumerate(scores_array):
        plt.text(coord[0], coord[1], n_range[i])
    plt.xlabel("MI(H, C)")
    plt.ylabel("$D_{KL} [P(W,C) || P(W,C|H)]$")
    plt.show()
