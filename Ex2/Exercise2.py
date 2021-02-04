
#===========================#
# Exercise 2: Methods in AI #
#===========================#

import numpy as np

# b)
def normalize(a):
    return a / np.sum(a)


def forward(Tmatrix,Omatrices,evidence):
    if not evidence:
        return np.ones(2)

    e = evidence.pop(-1)
    if e:
        return normalize(Omatrices[0] @ Tmatrix.T @ forward(Tmatrix, Omatrices, evidence.copy()))
    else:
        return normalize(Omatrices[1] @ Tmatrix.T @ forward(Tmatrix, Omatrices, evidence.copy()))


Omatrices = [np.array([[0.75, 0], [0, 0.2]]), np.array([[0.25, 0], [0, 0.8]])] # First: B_t = true,  Second: B_t = false
Tmatrix = np.array([[0.7, 0.3],[0.2, 0.8]])
evidence = [True, True, False, True, False, True]

print(forward(Tmatrix, Omatrices, evidence))

