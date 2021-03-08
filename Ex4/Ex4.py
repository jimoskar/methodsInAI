### Exercise 4 in Methods in AI

import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("test.csv")

xCat = df.loc[:, ['Pclass', 'Sex', 'Parch', 'Embarked']]
xCat[:] = xCat[:].astype("category")
y = df.loc[:, 'Survived']


print(xCat["Sex"].dtype)
print(xCat.to_numpy)
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


def plurVal(data):
    return data.value_counts().idxmax()

def allEqual(s):
    a = s.to_numpy() 
    return (a[0] == a).all()

def importance(a, e):
    pass


def DecisionTreeLearning(examples, attributes, response, parent_examples):
    '''Returns a decision tree based on examples'''
    if not examples: return plurVal(parent_examples[response])
    elif allEqual(examples[response]): examples[response][0]
    elif not attributes: return plurVal(examples[response])
    else:
        A = np.argmax(importance(attributes, examples))
        attributes.remove(A)
        tree = DecisionTree(A)
        for v in examples[A].cat.categories:
            exs = examples.loc[examples[A] == v] #e where e.A = v
            subtree = DecisionTreeLearning(exs,attributes, examples)
            tree.addBranch(v, subtree)
        return tree


    pass

tree = DecisionTreeClassifier(criterion="entropy")

tree.fit(xCat.to_numpy(), y)






'''
dot = Digraph(comment='The Round Table')

dot.node('A', 'A')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')

dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')

#dot.render('test-output/round-table.gv', view=True)
'''