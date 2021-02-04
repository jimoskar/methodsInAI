
#===========================#
# Exercise 2: Methods in AI #
#===========================#

import numpy as np

# b)
def normalize(a):
    return a / np.sum(a)


def forward(Tmatrix,Omatrices,evidence):
    if not evidence:
        return np.array([0.5, 0.5])

    e = evidence.pop(-1)
    if e:
        return Omatrices[0] @ Tmatrix.T @ forward(Tmatrix, Omatrices, evidence.copy())
    else:
        return Omatrices[1] @ Tmatrix.T @ forward(Tmatrix, Omatrices, evidence.copy())


def forward_bottomUp(Tmatrix, Omatrices, evidence):
    t = len(evidence)
    fList = [np.array([0.5, 0.5])]
    for i in range(t):
        e = evidence[i]
        f = None
        if e:
            f = normalize(Omatrices[0] @ Tmatrix.T @ fList[i])
        else:
            f = normalize(Omatrices[1] @ Tmatrix.T @ fList[i])
        fList.append(f)
    return fList

            





Omatrices = [np.array([[0.75, 0], [0, 0.2]]), np.array([[0.25, 0], [0, 0.8]])] # First: B_t = true,  Second: B_t = false
Tmatrix = np.array([[0.7, 0.3],[0.2, 0.8]])
evidence = [True, True, False, True, False, True]

print(normalize(forward(Tmatrix, Omatrices, evidence.copy())))
print(forward_bottomUp(Tmatrix,Omatrices, evidence))

# c)

def predict(Tmatrix, initial, tLower, tUpper):
    predList = []
    predList.append(initial @ Tmatrix)
    for k in range(tUpper - tLower):
        predList.append(predList[k] @ Tmatrix)
    return predList

initial = forward_bottomUp(Tmatrix,Omatrices,evidence)[-1]
print(initial)
print("\n")
print(predict(Tmatrix, initial, 7, 30))

    
