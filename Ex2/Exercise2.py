
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
#plt.show()

## e)


start_p = np.array([0.5, 0.5]) # Initial distribution over states
states = [0, 1] # 0 = 'no fish', 1 = 'fish'


def viterbi(Tmatrix, Omatrices, evidence, start_p, states):
    '''The Viterbi algorithm, based on pseudocode from Wikipedia.'''
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

best_path, T1, T2 = viterbi(Tmatrix, Omatrices, evidence, start_p, states)
print(T1)
print(T2)

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



def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


obs = ("birds", "birds", "noBirds", "birds", "birds", "noBirds")
states = ("fish", "noFish")
start_p = {"fish": 0.5, "noFish": 0.5}
trans_p = {
    "fish": {"fish": 0.8, "noFish": 0.2},
    "noFish": {"fish": 0.3, "noFish": 0.7},
}
emit_p = {
    "fish": {"birds": 0.75, "noBirds": 0.25},
    "noFish": {"birds": 0.2, "noBirds": 0.8},
}

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    best_st = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print ("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)


viterbi(obs, states, start_p, trans_p, emit_p)
