import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("mtcars.csv", usecols=(1, 2, 3, 4, 5, 6), delimiter=",", skiprows=1)

mpg = data[:, 0]
print("min mpg:", mpg.min())
print("max mpg:", mpg.max())
print("avg mpg:", mpg.mean())

six_cyl = data[:, 1] == 6
print("min mpg sa 6 cyl:", mpg[six_cyl].min())
print("max mpg sa 6 cyl:", mpg[six_cyl].max())
print("avg mpg sa 6 cyl:", mpg[six_cyl].mean())

plt.scatter(data[:, 0], data[:, 3], s=data[:, 5]*20, c="lime", edgecolor="k")

plt.show()
