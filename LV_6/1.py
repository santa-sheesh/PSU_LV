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

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='x')
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
