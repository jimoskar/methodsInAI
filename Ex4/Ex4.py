### Exercise 4 in Methods in AI

import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("test.csv")
#print(df.head())
#print(df['Pclass'])

#df['Sex'] = df['Sex'].astype("category")
xCat = df.loc[:, ['Pclass', 'Sex', 'Parch', 'Embarked']]
xCat["Sex"] = xCat["Sex"].astype("category")
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


def plurVal(data):
    return data.value_counts().idxmax()

def allEqual(s):
    a = s.to_numpy() 
    return (a[0] == a).all()

def importance(a, e):


def DecisionTreeLearning(examples, attributes, response, parent_examples):
    '''Returns a decision tree based on examples'''
    if not examples: return plurVal(parent_examples)
    elif allEqual(examples[response]): examples[response][0]
    elif not attributes: return plurVal(examples)
    else:
        A = np.argmax(importance(attributes, examples))
        tree = DecisionTree()
        tree.add(A)
        for v in A.values:
            exs = ... #e where e.A = v
            subtree = DecisionTreeLearning(exs,attributes - A, examples)
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