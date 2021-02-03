
#===========================#
# Exercise 2: Methods in AI #
#===========================#

import numpy as np

# b)
def normalize(a):
    return a / np.sum(a)


def forward(T,O,f):
    return normalize(O @ T.T @ f)

Omatrix = np.array([[0.75, 0], [0, 0.2]])
Tmatrix = np.array([[0.7, 0.3],[0.2, 0.8]])
