import numpy as np
import matplotlib.pyplot as plt


i = np.ones(2)
i.reshape(2,1)

print(np.shape(i))

print(np.argmax([i for i in range(3)]))


x = np.linspace(1,2)

plt.plot(x,x)
# plt.show()

a = np.arange(1,10)


for i in enumerate(a):
    print(i)


from scipy.interpolate import interp1d
from scipy.integrate import quad

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 10, num=41, endpoint=True)

import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

integrand = lambda x : f(x)**2
print(quad(integrand, 0, 10))
print(quad(f2, 0, 10))

