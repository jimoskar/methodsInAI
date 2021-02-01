import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,100)

def f(x):
    return np.cos(x)

print(x)
plt.plot(x,f(x))
plt.show()

