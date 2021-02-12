import numpy as np

i = np.ones(2)
i.reshape(2,1)

print(np.shape(i))

print(np.argmax([i for i in range(3)]))
