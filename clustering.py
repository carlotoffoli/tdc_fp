# Clustering analisys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from matplotlib import pyplot as plt
from util import error_plot
from preproc import filter_dataset, Preprocess

# pd.set_option('display.max_columns', None)

def algo_showcase(data: Preprocess, n_clusters: range, algorithm, head: float = 0.2, digits: int = 3):
    """
    Implements clustering secondo Martina

    @param head: number of clusters to be considered for feature analysis.
    (e.g. first 20%)
    @param n_clusters: range object - e.g. `range(4,20)`
    @param algorithm: the algorithm used for clustering, must return the labels as first param
    """

    for n in n_clusters:
        print(f"\n\nNumber of clusters: {n}")
        labels: np.array = algorithm(data.X, n)[0]

        results = []

        # For each n, print the highest risk clusters
        # To do that, analyze the single clusters
        for i in range(n):
            ith_cluster_patients = labels == i
            ith_patient_count = sum(ith_cluster_patients)
            # Actually the risk isn't normalized,
            # just scaled to the number of patients involved
            ith_risk = round(np.dot(data.Y, ith_cluster_patients) / ith_patient_count, digits)

            # Feature analysis
            feature_bounds = [
                (round(np.min(col), digits), round(np.max(col), digits), round(np.mean(col), digits))
                for col in filter_dataset(data.rescale(), lambda x: ith_cluster_patients[x[0]]).T
            ]
            
            results.append([i, ith_risk] + feature_bounds)
        
        results.sort(reverse=True, key=lambda cluster: cluster[1])

        print(f"Risk variance: %.2f" % np.array([i[1] for i in results]).var())

        # Generate DataFrame
        yield pd.DataFrame(results[:round(head*len(results))])

def kmeans(X: np.array, n_clusters = 2):
    cls = KMeans(n_clusters, n_init=1).fit(X)
    return cls.labels_, cls.cluster_centers_, cls.inertia_

def agglomerative(X: np.array, n_clusters = 2):
    cls = AgglomerativeClustering(n_clusters, linkage="ward").fit(X)
    return cls.labels_, cls.n_leaves_

def _elbow(X: np.array, n_max: int = None):
    """
    This function implements the elbow method to 
    select the optimal number of clusters in KMeans
    
    @param n_max: The maximum number of clusters. Defaults to n_samples
    """

    inertia = []
    end = X.shape[0] if not n_max else n_max
    for i in range(2, end):
        inertia.append(kmeans(X, i)[2])
    
    error_plot(inertia, range(2, end))

if __name__ == '__main__':
    from sklearn.metrics import precision_score
    from util import accuracy, plot_clusters, pca_decomposition
    from preproc import preprocessing

    X, Y, col_names = preprocessing()

    labels, _, _ = kmeans(X)

    print("Scores - also with inverted labels")
    print(f"Accuracy: {round(accuracy(labels, Y)*100, 2)} | {round((1-accuracy(labels, Y))*100, 2)} %")
    print(f"Precision: {round(precision_score(labels, Y)*100, 2)} | {round((1-precision_score(labels, Y))*100, 2)} %")
    # print(f"Inertia: {clustering.inertia_}")

    X_reduced, components = pca_decomposition(X)

    plot_clusters(X_reduced, (Y == labels).astype(int), "Correctly classified")
    plot_clusters(X_reduced, labels, "Cluster labels")
    plot_clusters(X_reduced, Y, "Dataset labels")

    print("PCA weights:")
    print(components[0])

    _elbow(X)
