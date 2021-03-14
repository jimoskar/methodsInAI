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
    def __init__(self, A, type = 'disc', split = None):
        self.label = A
        self.branches = dict()
        self.type = type
        self.split = split
    
    def addBranch(self, value, b):
        self.branches[value] = b

    def predict(self, observation):
        index = observation[self.label]
        if self.type == 'disc': # attribute is categorical/discrete
            if not isinstance(self.branches[index], DecisionTree):
                return self.branches[index]
            else:
                return self.branches[index].predict(observation)
        else: # attribute is continuous
            if index > self.split:
                index = '>' + str(self.split)
                if not isinstance(self.branches[index], DecisionTree):
                    return self.branches[index]
                else:
                    return self.branches[index].predict(observation)
            else:
                index = '<' + str(self.split)
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

        
    def plotTree(self, name):
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

        graph.render(name + '.gv', view=True)
            
        


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

    '''
    if a == 'SibSp':
            print("examples")
            print(exs)
    '''
    bestSplit = None
    bestRemainder = float('inf')
    for i in range(1,exs.shape[0]):
        remainder = 0
        split = None
        if (exs[res].iloc[i] != exs[res].iloc[i - 1]) and (exs[a].iloc[i] != exs[a].iloc[i - 1]):
            split = (exs[a].iloc[i] + exs[a].iloc[i -1 ])/2.0
            p1 = exs.loc[(exs[a] < split) & (exs[res] == 1)].shape[0]
            p2 = exs.loc[(exs[a] > split) & (exs[res] == 1)].shape[0]
            n1 = exs.loc[(exs[a] < split) & (exs[res] == 0)].shape[0]
            n2 = exs.loc[(exs[a] > split) & (exs[res] == 0)].shape[0]
            '''
            if a == 'SibSp':
                print("split")
                print(split)
                print(p1)
                print(p2)
                print(n1)
                print(n2)
             '''   
            if (p1 == 0 and n1 == 0)or (p2 == 0 and n2 == 0):
                continue
         
            
            remainder = (p1 + n1)/(p + n) * B(p1/(p1 + n1)) + (p2 + n2)/(p + n) * B(p2/(p2 + n2))
            if remainder < bestRemainder:
                bestRemainder = remainder
                bestSplit = split
    
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
        bestSplit = None
        if str(exs[a].dtype) != 'category':
            remainder, bestSplit = findBestSplit(a, res, exs, n, p)
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
            split = bestSplit

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
        
        if A == None:
            return plurVal(examples[response])

    
    
        attributes.remove(A)
        tree = None
        if str(examples[A].dtype) == 'category': # A is categorical
            tree = DecisionTree(A)
            for v in examples[A].cat.categories:
                exs = examples.loc[examples[A] == v] 
                attr = attributes.copy()
                subtree = DecisionTreeLearning(exs, attr, response, examples)
                tree.addBranch(v, subtree)
        else:

            tree = DecisionTree(A, 'cont', split)
            #print("split variable:")
            #print(A)
            exs1 = examples.loc[examples[A] > split] 
            attr1 = attributes.copy()
            subtree = DecisionTreeLearning(exs1, attr1, response, examples)
            tree.addBranch('<' + str(split), subtree)

            exs2 = examples.loc[examples[A] < split] 
            attr2 = attributes.copy()
            subtree = DecisionTreeLearning(exs2, attr2, response, examples)
            tree.addBranch('>' + str(split), subtree)

        return tree

def error(tree, test, attributes):
    '''Returns the test error.'''
    obs = test.loc[:, attributes]
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



train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')

def one_a():
    '''Assembles and plots the decision tree for 1a) and prints the test error.'''

    trainExamples = train.loc[:, ['Pclass', 'Sex', 'Embarked', 'Survived']]
    # Make the right attributes categorical:
    trainExamples.loc[:, ['Pclass', 'Sex', 'Embarked', 'Survived']] = trainExamples.loc[:, ['Pclass', 'Sex', 'Embarked', 'Survived']].astype("category")
    attributes = ['Pclass', 'Sex', 'Embarked']
    
    tree = DecisionTreeLearning(trainExamples, attributes.copy(), 'Survived')
    tree.plotTree('one_a')

    print("Test error rate: ")
    print(error(tree, test, attributes))


def one_b():
    '''Assembles and plots the decision tree for 1b) and prints the test error.'''

    trainExamples = train.loc[:, ['Pclass', 'Sex', 'Embarked', 'Fare', 'SibSp', 'Parch', 'Survived']]
    # Make the right attributes categorical:
    trainExamples.loc[:, ['Pclass', 'Sex', 'Embarked', 'Survived']] = trainExamples.loc[:, ['Pclass', 'Sex', 'Embarked', 'Survived']].astype("category")
    attributes = ['Pclass', 'Sex', 'Fare', 'Embarked', 'SibSp', 'Parch']
    
    tree = DecisionTreeLearning(trainExamples, attributes.copy(), 'Survived')
    tree.plotTree('one_b')

    print("Test error rate:")
    print(error(tree, test, attributes))


def restaurantExample():
    '''Tests the model on the restaurant example from the book'''

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
    tree = DecisionTreeLearning(restaurant, ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'], 'WillWait')
    tree.plotTree('restaurant')


#===========================#
# Test the code under here! #
#===========================#

#one_a()
#restaurantExample()
one_b()



