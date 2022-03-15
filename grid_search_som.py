import multiprocessing
import os.path
import pickle
import sys

from convergence import evaluate_convergence_model
from som import SelfOrganisingMap, product_dict


def func(args):
    som_args = args[0]
    language_ids = args[1]
    sample_range_ = args[2]
    seed_ = args[3]

    print(som_args)
    print(language_ids)
    print(sample_range_)
    print(seed_)

    som = SelfOrganisingMap(**som_args)
    som.learn_languages(
        sample_range_[-1],
        scoring_steps=None,
        language_ids=language_ids,
        seed=seed_,
    )
    acc = evaluate_convergence_model(som, language_ids=language_ids)
    return som_args, acc


if __name__ == '__main__':
    seed = 42
    workers = 12
    lids = [2]  # list(range(1, 111))
    path = f"output/som/{seed}/grid_search"

    if not os.path.exists(path):
        os.mkdir(path)

    features = {
        "features": ["xling", "perc"],
        "sampling": ["corpus"],
        "sigma": [5.0],
        "term_weight": [0.1, 0.3, 0.5],
        "alpha": [0.1, 0.3, 0.5],
        "size": {7, 10, 12},
        "color_prior": ["capacity"],
    }

    # Number of samples to draw for each language
    sample_range = (
            list(range(1, 25, 1))
            + list(range(25, 50, 5))
            + list(range(50, 100, 10))
            + list(range(100, 220, 20))
            + list(range(250, 1000, 50))
            + list(range(1000, 2100, 100))
            + list(range(3000, 10001, 1000))
            + list(range(20000, 100001, 10000))
    )

    pargs = [(args, lids, sample_range, seed) for args in product_dict(**features)]

    i = int(sys.argv[1])
    args_accs = func(pargs[i])

    pickle.dump(args_accs, open(os.path.join(path, f"{i}.p"), "wb"))
