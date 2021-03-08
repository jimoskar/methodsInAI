import pandas as pd
import numpy as np


def allEqual(s):
    a = s.to_numpy() 
    return (a[0] == a).all()

def plurVal(data):
    return data.value_counts().idxmax()    

ds = pd.DataFrame({'A':[1,1,3], 'B':[1,1,1]})

df = pd.read_csv("test.csv")
#print(df.head())
#print(df['Pclass'])

#df['Sex'] = df['Sex'].astype("category")
xCat = df.loc[:, ['Pclass', 'Sex', 'Parch', 'Embarked']]

xCat[:] = xCat[:].astype("category")
l = xCat["Pclass"].cat.categories
for e in l:
    print(e)


print(ds.loc[ds['A'] == 1])
