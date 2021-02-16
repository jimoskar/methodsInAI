
#===========================#
# Exercise 2: Methods in AI #
#===========================#
# Note to reader: I have sticked to the convention that row 0 or index 0 corresponds to the state false, and row 1 
# or index 1 corresponds to the state false. This can be a bit confusing with reference to the course book by Russel
# and Norvig.

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


## b)
def normalize(a):
    '''Normalize distribution a.'''
    return a / np.sum(a)



def forward(Tmatrix, Omatrices, evidence):
    '''Compute a list of forward messages given evidence.'''
    t = len(evidence)
    fList = [np.array([0.5, 0.5])]
    for i in range(t):
        e = evidence[i]
        f = None
        if e:
            f = normalize(Omatrices[e] @ Tmatrix.T @ fList[i])
        else:
            f = normalize(Omatrices[e] @ Tmatrix.T @ fList[i])
        fList.append(f)
    return fList

            

Omatrices = [np.array([[0.8, 0], [0, 0.25]]), np.array([[0.2, 0], [0, 0.75]])] # First: no birds,  Second: birds nearby
Tmatrix = np.array([[0.7, 0.3],[0.2, 0.8]]) # first row corresponds to Xt = false and second row corresponds to Xt = true
evidence = [1, 1, 0, 1, 0, 1]

filter = forward(Tmatrix,Omatrices, evidence)
print("\nFiltering: ")
print(filter)

# Plotting:
x = [i for i in range(len(filter))]
p = [f[1] for f in filter]
plt.plot(x,p, label = "Filtering", linewidth = 3)
plt.xlabel("Time, $t$")
plt.ylabel("Probability")

## c)

def predict(Tmatrix, initial, tLower, tUpper):
    '''Compute the predicted distribution of the state variable up until time tUpper'''
    predList = []
    predList.append(initial @ Tmatrix)
    for k in range(tUpper - tLower):
        predList.append(predList[k] @ Tmatrix)
    return predList


initial = forward(Tmatrix,Omatrices,evidence)[-1]
prediction = predict(Tmatrix, initial, 7, 30)
print("\n Prediction:")
print(prediction)

# Plotting:
y = [p[1] for p in prediction]
x = [i for i in range(7,31)]
plt.plot(x,y, label = "Prediction", linewidth = 3)

## d)

def backward(Tmatrix, Omatrices, evidence):
    '''Compute a list of the backward messages given evidence.'''
    b_list = [np.ones(2)]
    for i in range(len(evidence) - 1,-1,-1):
        e = evidence[i]
        b = Tmatrix @ Omatrices[e] @ b_list[-1]
        b_list.append(b)

    b_list.reverse()   
    return b_list



def smooth(Tmatrix, Omatrices, evidence):
    '''Compute the smoothed distributions of the state varable given new evidence.'''
    fm = forward(Tmatrix, Omatrices, evidence)
    bm = backward(Tmatrix, Omatrices, evidence)
    sv = []

    for i in range(len(evidence)):
        s = normalize(fm[i] * bm[i])
        sv.append(s)

    return sv

smoothing = smooth(Tmatrix, Omatrices, evidence)
print("\nSmoothing: ")
print(smoothing)

# Plotting:
x = [i for i in range(0, 6)]
y = [s[1] for s in smoothing]
plt.plot(x, y, label = "Smoothing", linestyle = "dashed", linewidth = 3)
plt.legend()
plt.show()

## e)


start_p = np.array([0.5, 0.5]) # Initial distribution over states
states = [0, 1] # 0 = 'no fish', 1 = 'fish'


def viterbi(Tmatrix, Omatrices, evidence, start_p, states):
    '''The Viterbi algorithm, based on pseudocode from Wikipedia.
       Return the best path and  two tables:
       T1 contains an unscaled version of m1:t from the book
       T2 contains pointers to the most likely previous state for each state
       REMEMBER: row 1 corresponds to Xt = true.
    '''
    T1 = np.zeros((len(states), len(evidence)))
    T2 = np.zeros((len(states), len(evidence)))
    for s in range(len(states)):
        T1[s,0] = start_p[s] * Omatrices[evidence[0]][s,s]
    
    for o in range(1, len(evidence)):
        for s in range(len(states)):
            k = np.argmax([T1[k, o - 1] * Tmatrix[k, s] * Omatrices[evidence[o]][s, s] for k in range(len(states))])
            T1[s, o] = T1[k, o  - 1] * Tmatrix[k, s] * Omatrices[evidence[o]][s, s]
            T2[s, o] = k
    best_path  = []
    for o in range(-1, -(len(evidence) + 1), -1):
        k = np.argmax(T1[:, o])
        best_path.insert(0, states[k])
    return best_path, T1, T2


for t in range(len(evidence)):
    e = evidence[:(t+1)]
    print("\nViterbi for t = " + str(t+1))
    best_path, T1, T2 = viterbi(Tmatrix, Omatrices, e, start_p, states)
    print("probs: ")
    print(T1)
    print("pointers:")
    print(T2)


# Try umbrella example:

Omatrices = [np.array([[0.8, 0], [0, 0.1]]), np.array([[0.2, 0], [0, 0.9]])] # First: no birds,  Second: birds nearby
Tmatrix = np.array([[0.7, 0.3],[0.3, 0.7]]) # first row corresponds to Xt = false and second row corresponds to Xt = true
evidence = [1, 1, 0, 1, 1]
start_p = np.array([0.5,0.5])
best_path, T1, T2 = viterbi(Tmatrix, Omatrices, evidence, start_p, states)
print("\n Umbrella:")
print(best_path)
print(T1)
print(T2)

# Seems to give the correct results!

