import numpy as np
from scipy.sparse import csc_matrix

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
zxc = csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()

print(zxc)
