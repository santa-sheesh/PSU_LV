import numpy as np
import matplotlib.pyplot as plt

def check(size, rows, cols):
    black = np.zeros((size, size))
    white = np.ones((size, size)) * 255
    row1 = np.hstack([(black if j % 2 == 0 else white) for j in range(cols)])
    row2 = np.hstack([(white if j % 2 == 0 else black) for j in range(cols)])
    board = np.vstack([(row1 if i % 2 == 0 else row2) for i in range(rows)])
    return board

img = check(100, 4, 5)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.show()
