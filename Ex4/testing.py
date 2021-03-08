import pandas as pd
import numpy as np


def allEqual(s):
    a = s.to_numpy() 
    return (a[0] == a).all()

df = pd.DataFrame({'A':[1,2,3], 'B':[1,1,1]})

df = pd.read_csv("test.csv")
#print(df.head())
#print(df['Pclass'])

#df['Sex'] = df['Sex'].astype("category")
xCat = df.loc[:, ['Pclass', 'Sex', 'Parch', 'Embarked']]

xCat[:] = xCat[:].astype("category")
print(xCat["Pclass"].cat.categories)

