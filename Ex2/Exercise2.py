
#===========================#
# Exercise 2: Methods in AI #
#===========================#

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# b)
def normalize(a):
    return a / np.sum(a)



def forward(Tmatrix, Omatrices, evidence):
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
print("filter probs: ")
print(filter)
x = [i for i in range(len(filter))]
p = [f[1] for f in filter]

plt.plot(x,p, label = "Filtering", linewidth = 3)
plt.xlabel("Time, $t$")
plt.ylabel("Probability")
#plt.show()

# c)

def predict(Tmatrix, initial, tLower, tUpper):
    predList = []
    predList.append(initial @ Tmatrix)
    for k in range(tUpper - tLower):
        predList.append(predList[k] @ Tmatrix)
    return predList


initial = forward(Tmatrix,Omatrices,evidence)[-1]
print("predict:")
prediction = predict(Tmatrix, initial, 7, 30)
print(prediction)

y = [p[1] for p in prediction]
x = [i for i in range(7,31)]
plt.plot(x,y, label = "Prediction", linewidth = 3)

# This operation is called prediction, beacuse it predicts the distribution og X's states for future time points t > k,
# for evidence given up until t = k.
# As t increases, we can see that the the distribution over X apporaches a stationary distribution.

# d)

def backward(Tmatrix, Omatrices, evidence):
    b_list = [np.ones(2)]
    for i in range(len(evidence) - 1,-1,-1):
        e = evidence[i]
        b = Tmatrix @ Omatrices[e] @ b_list[-1]
        b_list.append(b)

    b_list.reverse()   
    return b_list



def smooth(Tmatrix, Omatrices, evidence):
    fm = forward(Tmatrix, Omatrices, evidence)
    bm = backward(Tmatrix, Omatrices, evidence)
    sv = []

    for i in range(len(evidence)):
        s = normalize(fm[i] * bm[i])
        sv.append(s)

    return sv

smoothing = smooth(Tmatrix, Omatrices, evidence)
print("smooth: ")
print(smoothing)

x = [i for i in range(0, 6)]
y = [s[1] for s in smoothing]
plt.plot(x, y, label = "Smoothing", linestyle = "dashed", linewidth = 3)
plt.legend()
plt.show()


# This is called smoothing

# e)


start_p = np.array([0.5, 0.5]) # Initial distribution over states
states = [0, 1] # 0 = 'no fish', 1 = 'fish'


def viterbi(Tmatrix, Omatrices, evidence, start_p, states):
    V = np.zeros((len(states), len(evidence)))
    for s in range(len(states)):
        V[s,0] = start_p[s] * Omatrices[evidence[0]][s,s]

    for o in range(1, len(evidence)):
        for s in range(len(states)):
            k = np.argmax([V[k, o - 1] * Tmatrix[k, s] * Omatrices[evidence[o]][s, s] for k in range(len(states))])
            V[s, o] = V[k, o  - 1] * Tmatrix[k, s] * Omatrices[evidence[o]][s, s]
    
    best_path  = []
    print(V)
    for o in range(-1, -(len(evidence) + 1), -1):
        k = np.argmax(V[:, o])
        best_path.insert(0, states[k])
    return best_path

print("Viterbi: ")
print(viterbi(Tmatrix, Omatrices, evidence, start_p, states))



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
