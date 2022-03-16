from ast import literal_eval
import os
import pickle

seed = 42
path = f"output/som/{seed}/grid_search"

gs = {}
for i in range(54):
    args, accs = pickle.load(open(os.path.join(path, f"{i}.p"), "rb"))
    gs[str(args)] = accs

results = {}
result_params = {}
for som_params, evals in gs.items():
    if "xling" in som_params: continue
    for lid, acc in evals.items():
        if lid not in results or results[lid] < acc:
            results[lid] = acc
            result_params[lid] = literal_eval(som_params)

pickle.dump(result_params, open(os.path.join(path, "grid_search_params.p"), "wb"))
