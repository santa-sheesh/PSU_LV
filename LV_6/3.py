from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

def plot_dendrogram(X, method):
    Z = linkage(X, method=method)
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title(f'Dendrogram (method={method})')
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.show()

for m in ['single', 'complete', 'average', 'ward']:
    plot_dendrogram(X, m)
