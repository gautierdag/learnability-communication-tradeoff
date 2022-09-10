import random
import numpy as np

SEED = 42

random.seed(SEED)
np.seterr(divide="ignore")
np.random.seed(SEED)

N = 15  # Number of terms to use
grid = np.zeros((8, 40), dtype=int) - 1  # The Munsell chip-grid without the 10 graysale chips

shuffle = np.arange(grid.size)
np.random.shuffle(shuffle)

idx = np.unravel_index(shuffle[:N], grid.shape)
grid[idx] = np.arange(N)

while np.any(grid == -1):
    for i in shuffle:
        idx = np.unravel_index(i, grid.shape)
        if grid[idx] == -1:
            u, v = idx
            subset = grid[max(0, u - 1):min(u + 2, grid.shape[0]), max(0, v - 1):min(v + 2, grid.shape[1])]
            if np.all(subset == -1):
                continue
            vals, counts = np.unique(subset, return_counts=True)
            vals, counts = vals[1:], counts[1:]  # Drop counts for -1
            probs = counts / counts.sum()
            grid[idx] = np.random.choice(vals, 1, p=probs)
