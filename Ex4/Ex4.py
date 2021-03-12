### Exercise 4 in Methods in AI

import pandas as pd
import numpy as np
from graphviz import Digraph
from pandas.core.series import Series
from sklearn.tree import DecisionTreeClassifier
import math





#print(xCat.head())
# Continuous: Age, Fare
# Irrelevant: Name, Cabin, Ticket

class DecisionTree:
    def __init__(self, A):
        self.label = A
        self.branches = dict()
    
    def addBranch(self, value, b):
        self.branches[value] = b

    def predict(self, observation):
        print(observation)
        index = observation[self.label]
        if not isinstance(self.branches[index], DecisionTree):
            return self.branches[index]
        else:
            return self.branches[index].predict(observation)

    def printTree(self):
        print(str(self.label) + ": [")
        for key, value in self.branches.items():
            if not isinstance(value, DecisionTree):
                print(value)
            else:
                value.printTree()
        print("]")
    
    def addToPlot(self, parent, edge, graph, ids):
        parentId = ids.pop()
        graph.node(str(parentId), self.label)
        graph.edge(str(parent), str(parentId), label = str(edge))
        for key, value in self.branches.items():
            if not isinstance(value, DecisionTree):
                id = ids.pop()
                graph.node(str(id), str(value))
                graph.edge(str(parentId), str(id), label = str(key))
            else:
                value.addToPlot(parentId, key, graph, ids)

        
    def plotTree(self):
        ids = [i for i in range(100)]
        parentId = ids.pop()
        graph = Digraph(comment='Decision Tree')
        graph.node(str(parentId), self.label)
        for key, value in self.branches.items():
            if not isinstance(value, DecisionTree):
                id = ids.pop()
                graph.node(str(id), str(value))
                graph.edge(str(parentId), str(id), label = str(key))
            else:
                value.addToPlot(parentId, key, graph, ids)

        graph.render('test-output/round-table.gv', view=True)
            
        


def plurVal(data):
    return data.value_counts().idxmax()

def allEqual(s):
    a = s.to_numpy() 
    return (a[0] == a).all()

def B(q):
    '''Entropy of a boolean variable'''
    if q == 1 or q == 0: 
        return 0
    return -( q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2))

def findBestSplit(a, res, exs, p, n):
    exs = exs.sort_values(by = [a])
    bestSplit = None
    bestRemainder = float('inf')
    for i in range(1,exs.shape[0]):
        remainder = 0
        if exs[res][i] != exs[res][i - 1]:
            p1 = exs.loc[(exs[a] >= exs[a][i]) & (exs[res][i] == 1)].shape[0]
            p2 = exs.loc[(exs[a] < exs[a][i]) & (exs[res][i] == 1)].shape[0]
            n1 = exs.loc[(exs[a] >= exs[a][i]) & (exs[res][i] == 0)].shape[0]
            n2 = exs.loc[(exs[a] < exs[a][i]) & (exs[res][i] == 0)].shape[0]
            remainder = (p1 + n1)/(p + n) * B(p1/(p1 + n1)) + (p2 + n2)/(p + n) * B(p2/(p2 + n2))
            if remainder < bestRemainder:
                bestRemainder = remainder
                bestSplit = exs[a][i]
    
    return bestRemainder, bestSplit









def importance(atr, exs, res):
    vals = exs[res].value_counts()
    p = vals[1]
    n = vals[0]
    ent = B(p/(p + n))
    
    bestGain = float('-inf')
    bestAtr = None
    split = None
    for a in atr:
        remainder = 0
        print(str(exs[a].dtype))
        if str(exs[a].dtype) != 'category':
            remainder, split = findBestSplit(a, res, exs, n, p)
        else:
            counts = exs.groupby(res)[a].value_counts().unstack(fill_value=0).stack()
            for v in exs[a].cat.categories:
                if v not in counts.index.levels[1]:
                    remainder += 0
                else:
                    pk = counts[1, v]
                    nk = counts[0, v]
                    remainder += (pk + nk)/(p + n) * B(pk/(pk + nk))

        gain = ent - remainder
        if gain > bestGain:
            bestGain = gain
            bestAtr = a

    return bestAtr, split




def DecisionTreeLearning(examples, attributes, response, parent_examples = None):
    '''Returns a decision tree based on examples'''
    if examples.empty: 
        return plurVal(parent_examples[response])
    elif allEqual(examples[response]): 
        return examples[response].iloc[0]
    elif not attributes: 
        return plurVal(examples[response])
    else:
        A, split = importance(attributes, examples, response)
        attributes.remove(A)
        tree = DecisionTree(A)
        if split == None: # A is categorical
            for v in examples[A].cat.categories:
                exs = examples.loc[examples[A] == v] 
                attr = attributes.copy()
                subtree = DecisionTreeLearning(exs, attr, response, examples)
                tree.addBranch(v, subtree)
        else:
            
            exs1 = examples.loc[examples[A] > split] 
            attr1 = attributes.copy()
            subtree = DecisionTreeLearning(exs1, attr1, response, examples)
            tree.addBranch('>=' + str(split), subtree)

            exs2 = examples.loc[examples[A] < split] 
            attr2 = attributes.copy()
            subtree = DecisionTreeLearning(exs2, attr2, response, examples)
            tree.addBranch('<' + str(split), subtree)

        return tree



train = pd.read_csv("train.csv")

X = train.loc[:, ['Pclass', 'Sex', 'Embarked', 'Fare', 'Survived']]
X.loc[:, ['Pclass', 'Sex', 'Embarked', 'Survived']] = X.loc[:, ['Pclass', 'Sex', 'Embarked', 'Survived']].astype("category")



attributes = ['Pclass', 'Sex', 'Fare', 'Embarked']
tree = DecisionTreeLearning(X, attributes.copy(), 'Survived')
#tree.printTree()
#tree.plotTree()

# Testing

test = pd.read_csv('test.csv')

def error(tree, test, attributes):
    obs = test.loc[:, attributes]
    print(obs)
    resp = test.loc[:, 'Survived']
    n = resp.shape[0]
    t = 0
    f = 0
    for i in range(n):
        pred = tree.predict(obs.iloc[i])
        if pred == resp.iloc[i]:
            t += 1
        else:
            f += 1
    
    return t/(t + f)

print("at")
print(attributes)
print(error(tree, test, attributes))


'''

restaurant = pd.DataFrame({'Alt' : [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1], 
                            'Bar' : [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
                            'Fri' : [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                            'Hun' : [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            'Pat' : ['S', 'F', 'S', 'F', 'F', 'S', 'N', 'S', 'F', 'F', 'N', 'F'],
                            'Price' : ['$$$', '$', '$', '$', '$$$', '$$', '$', '$$', '$', '$$$', '$', '$'],
                            'Rain' : [0, 0, 0, 1 , 0, 1, 1, 1, 1, 0, 0, 0],
                            'Res' : [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                            'Type': ['F', 'T', 'B', 'T', 'F', 'I', 'B', 'T', 'B', 'I', 'T', 'B'],
                            'Est' : [0, 1, 0, 1, 2, 0, 0, 0, 2, 1, 0, 2],
                            'WillWait' : [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]})


restaurant[:] = restaurant[:].astype("category")
print(restaurant)
print(restaurant.groupby('WillWait')['Alt'].value_counts().unstack(fill_value=0).stack())
tree = DecisionTreeLearning(restaurant, ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'], 'WillWait')
tree.plotTree()
'''