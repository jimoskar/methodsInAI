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
    
