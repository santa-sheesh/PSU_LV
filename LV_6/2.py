from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def generate_data(n_samples, method):
    if method == 1:
        X, _ = datasets.make_blobs(n_samples=n_samples, random_state=365)
    elif method == 2:
        X, _ = datasets.make_blobs(n_samples=n_samples, random_state=148)
        X = X @ [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    elif method == 3:
        X, _ = datasets.make_blobs(n_samples=n_samples, centers=4,
                                   cluster_std=[1.0, 2.5, 0.5, 3.0],
                                   random_state=148)
    elif method == 4:
        X, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    elif method == 5:
        X, _ = datasets.make_moons(n_samples=n_samples, noise=0.05)
    else:
        X = np.array([])
    return X

X = generate_data(500, 1)

inertia = [KMeans(n_clusters=k, random_state=0).fit(X).inertia_ for k in range(1, 21)]

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), inertia, 'bo-')
plt.xlabel("Broj klastera")
plt.ylabel("Inercija")
plt.title("Elbow metoda za optimalan broj klastera")
plt.grid(True)
plt.show()
