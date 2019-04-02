import numpy as np
from numpy import matmul as mm
from np.random import randint, randn
from np.linalg import svd
from scipy.io import imread
import matplotlib.pyplot as plt

def normalize_columns(X):
    return mm(X, np.diag(1./(np.sqrt(np.sum(a**2, axis=0))+1e-6)))

def get_batch(X, patchshape, batch_size=200):
    ix = randint(0,X.shape[1]-patchshape[0])
    iy = randint(0,X.shape[2]-patchshape[1])
    iz = randint(0,X.shape[3]-patchshape[2]) if patchshape[2] != X.shape[3] else 0
    B = X[randint(0,X.shape[0], size=batch_size),ix:ix + patchshape[0],iy:iy + patchshape[1],iz:iz + patchshape[2]].reshape(batch_size, patchshape[0]*patchshape[1]*patchshape[2])
    B = B.T
    B = B - np.mean(B, axis=0)
    U,S,V = svd(mm(B, B.T))
    ZCAMatrix = mm(U, mm(np.diag(1.0/np.sqrt(S+1e-5)),U.T))
    B = mm(B, ZCAMatrix)
    return B

def train_XCubed(W, dataset, patchshape, batch_size, iterations):
    for i in range(iterations):
        batch = get_batch(X, patchshape, batch_size=batch_size)
        W = normalize_columns(W)
        a = 0.5*normalize_columns(mm(W.T, batch))**3
        W += mm((batch - mm(W,a)),a.T)
    return W

import glob
filelist = glob.glob('data/*.png')
Data = np.array([imread(fname) for fname in filelist])

Filters = randn(363,64)
Filters = train_XCubed(Filters, Data, (11,11,3), 200, 1000)
Filters = Filters.transpose(1,0).reshape(64,11,11,3)
np.save('filters.npy', Filters)

fig, axs = plt.subplots(8, 8)
fig.suptitle('Features')
plt.axes('off')
for i in range(8):
    for j in range(8):
        axs[i][j].imshow(Filters[i*8+j])
plt.savefig('filters.png')
plt.show()