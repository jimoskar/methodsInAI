### Exercise 4 in Methods in AI

import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("test.csv")
print(df.head())
print(df['Pclass'])

xCat = df.loc[:, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]
print(xCat.head())
# Continuous: Age, Fare
# Irrelevant: Name, Cabin, 

class Node:
    pass

class DecisionTree:
    pass

def DecisionTreeLearning(examples, attributes, parent_examples):
    '''Returns a decision tree based on examples'''

    pass

dot = Digraph(comment='The Round Table')

dot.node('A', 'A')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')

dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')

#dot.render('test-output/round-table.gv', view=True)
