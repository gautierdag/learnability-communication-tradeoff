import multiprocessing
import os.path
import pickle
import sys

from convergence import evaluate_convergence_model
from som import SelfOrganisingMap, product_dict, sample_range


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
    lids = list(range(1, 111))
    path = f"output/som/{seed}/grid_search"

    if not os.path.exists(path):
        os.mkdir(path)

    features = {
        "features": ["perc"],
        "sampling": ["corpus"],
        "sigma": [5.0],
        "term_weight": [0.01, 0.025, 0.075, 0.1],
        "alpha": [0.01, 0.025, 0.075, 0.1],
        "size": {7, 9, 11, 13},
        "color_prior": ["capacity"],
    }

    pargs = [(args, lids, sample_range, seed) for args in product_dict(**features)]

    i = int(sys.argv[1])
    args_accs = func(pargs[i])

    pickle.dump(args_accs, open(os.path.join(path, f"{i}.p"), "wb"))
