import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 3, 3, 1, 3])
y = np.array([1, 2, 2, 1, 1, 1, 1])

plt.plot(x, y, 'bo-', linewidth=2)
plt.xlabel("x os")
plt.ylabel("y os")

plt.xlim(0, 4)
plt.ylim(0, 4)
plt.grid(True)

plt.show()
