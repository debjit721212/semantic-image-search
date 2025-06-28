import numpy as np
from config import CACHE_PATH

data = np.load(CACHE_PATH, allow_pickle=True)
print(data['paths'])