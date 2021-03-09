### Exercise 4 in Methods in AI

import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn.tree import DecisionTreeClassifier
import math


df = pd.read_csv("test.csv")

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
        if not self.dict: return self.label
        return self.branches[observation[self.A]].predict(observation)

    def printTree(self):
        print(self.branches)
    
    def addToPlot(self, parent, edge, graph):
        graph.node(self.label)
        graph.edge(parent, self.label, label = str(edge))
        for key, value in self.branches.items():
            if not isinstance(value, DecisionTree):
                graph.node(str(key), str(value))
                graph.edge(self.label, str(key), label = str(key))
            else:
                value.addToPlot(self.label, key, graph)

        '''
dot = Digraph(comment='The Round Table')

dot.node('A', 'A')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')

dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')

#dot.render('test-output/round-table.gv', view=True)
'''
        
    def plotTree(self):
        graph = Digraph(comment='Decision Tree')
        graph.node(self.label, self.label)
        for key, value in self.branches.items():
            if not isinstance(value, DecisionTree):
                graph.node(str(key), str(value))
                graph.edge(self.label, str(key), label = str(key))
            else:
                value.addToPlot(self.label, key, graph)

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
    if examples.empty: return plurVal(parent_examples[response])
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
            subtree = DecisionTreeLearning(exs, attributes, response, examples)
            tree.addBranch(v, subtree)
        return tree


attributes = ['Pclass', 'Sex', 'Embarked']
tree = DecisionTreeLearning(xCat, attributes, 'Survived')
tree.printTree()
tree.plotTree()




#tree = DecisionTreeClassifier(criterion="entropy")

#tree.fit(xCat.to_numpy(), y)