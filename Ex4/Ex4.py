### Exercise 4 in Methods in AI

import pandas as pd
import numpy as np
from graphviz import Digraph
from pandas.core.series import Series
from sklearn.tree import DecisionTreeClassifier
import math


df = pd.read_csv("train.csv")

xCat = df.loc[:, ['Pclass', 'Sex', 'Embarked', 'Survived']]
xCat[:] = xCat[:].astype("category")
y = df.loc[:, 'Survived']


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


def importance(atr, exs, res):
    vals = exs[res].value_counts()
    p = vals[1]
    n = vals[0]
    ent = B(p/(p + n))
    
    bestGain = float('-inf')
    bestAtr = None
    for a in atr:
        remainder = 0
        counts = df.groupby(res)[a].value_counts().unstack(fill_value=0).stack()

        for v in exs[a].cat.categories:
            pk = counts[1, v]
            nk = counts[0, v]
            remainder += (pk + nk)/(p + n) * B(pk/(pk + nk))

        gain = ent - remainder
        if gain > bestGain:
            bestGain = gain
            bestAtr = a

    return bestAtr




def DecisionTreeLearning(examples, attributes, response, parent_examples = None):
    '''Returns a decision tree based on examples'''
    if examples.empty: 
        return plurVal(parent_examples[response])
    elif allEqual(examples[response]): 
        print("heyo") 
        print(examples[response].iloc[0])
        return examples[response].iloc[0]
    elif not attributes: 
        return plurVal(examples[response])
    else:
        A = importance(attributes, examples, response)
        attributes.remove(A)
        tree = DecisionTree(A)
        for v in examples[A].cat.categories:
            exs = examples.loc[examples[A] == v] 
            attr = attributes.copy()
            subtree = DecisionTreeLearning(exs, attr, response, examples)
            tree.addBranch(v, subtree)
        return tree


attributes = ['Pclass', 'Sex', 'Embarked']
tree = DecisionTreeLearning(xCat, attributes, 'Survived')
tree.printTree()
tree.plotTree()

#print("prediction:")
#print(tree.predict(obs))

# Testing

test = pd.read_csv('test.csv')

def error(tree, test):
    obs = test.loc[:, ['Pclass', 'Sex', 'Embarked']]
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

print(error(tree, test))




#tree = DecisionTreeClassifier(criterion="entropy")

#tree.fit(xCat.to_numpy(), y)