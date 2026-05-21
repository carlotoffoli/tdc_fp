# Clustering analisys
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from matplotlib import pyplot as plt

def algo_showcase():
    ...

def kmeans(X: np.array, n_clusters = 2):
    cls = KMeans(n_clusters, n_init=10).fit(X)
    return cls.labels_, cls.cluster_centers_, cls.inertia_

def agglomerative(X: np.array, n_clusters = 2):
    cls = AgglomerativeClustering(n_clusters, linkage="ward").fit(X)
    return cls.labels_, cls.n_leaves_, cls.distances_

def _elbow(X: np.array, n_max: int = None):
    """
    This function implements the elbow method to 
    select the optimal number of clusters in KMeans
    
    @param n_max: The maximum number of clusters. Defaults to n_samples
    """

    inertia = []
    for i in range(1, X.shape[0] if not n_max else n_max):
        inertia.append(kmeans(X)[2])
    # TODO: plot the inertia

if __name__ == '__main__':
    from sklearn.metrics import precision_score
    from util import accuracy, plot_clusters, pca_decomposition

    # TODO: Define X and Y

    labels, _, _ = kmeans(X)

    print("Scores - also with inverted labels")
    print(f"Accuracy: {round(accuracy(labels, Y)*100, 2)} | {round((1-accuracy(labels, Y))*100, 2)} %")
    print(f"Precision: {round(precision_score(labels, Y)*100, 2)} | {round((1-precision_score(labels, Y))*100, 2)} %")
    # print(f"Inertia: {clustering.inertia_}")

    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    plot_clusters(X_reduced, (Y == labels).astype(int), "Correctly classified")
    plot_clusters(X_reduced, labels, "Cluster labels")
    plot_clusters(X_reduced, Y, "Dataset labels")

    print("PCA weights:")
    print(pca.components_[0])