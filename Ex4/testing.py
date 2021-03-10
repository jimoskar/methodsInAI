import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


def allEqual(s):
    a = s.to_numpy() 
    return (a[0] == a).all()

def plurVal(data):
    return data.value_counts().idxmax()    

ds = pd.DataFrame({'A':[3,1,3], 'B':[1,1,1]})

df = pd.read_csv("test.csv")
#print(df.head())
#print(df['Pclass'])

#df['Sex'] = df['Sex'].astype("category")
xCat = df.loc[:, ['Pclass', 'Sex', 'Parch', 'Embarked']]

xCat[:] = xCat[:].astype("category")
l = xCat["Pclass"].cat.categories
for e in l:
    #print(e)
    pass

vals = ds['A'].value_counts()


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.loc[:,['Pclass', 'Sex', 'Embarked']]
X[:] = X[:].astype("category")
X['Sex'] = X['Sex'].cat.codes
X['Embarked'] = X['Embarked'].cat.codes
X = X.to_numpy()
print(X)
Y = train['Survived']
Y = Y.to_numpy()
print(Y)

tr = tree.DecisionTreeClassifier(criterion = 'entropy')
fit = tr.fit(X, Y)

#print(tree.plot_tree(fit))
#print("hei")

import graphviz 
dot_data = tree.export_graphviz(fit, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render()
graph.view()







