dict = {"a":1, "b":2}

for e in dict.items() :
    print(e)


a_subset = {key: value for key, value in dict.items() if value >= 2}
print(a_subset)

s = {1,2,3}

l = {1,2}

if l.issubset(s):
    print("true")
    print(l.union(s))
    
l = [1,2,3]
print(l[:1])

import numpy as np

a = np.zeros(2)

s = {1,2,3}

print(1 in s)

