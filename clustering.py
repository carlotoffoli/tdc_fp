# Clustering analisys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from util import error_plot
from preproc import filter_dataset, Preprocess

def algo_showcase(data: Preprocess, n_clusters: range, algorithm, head: float = 0.2, digits: int = 3):
    """
    Implements clustering according to Martina

    @param head: number of clusters to be considered for feature analysis.
    (e.g. first 20%)
    @param n_clusters: range object - e.g. `range(4,20)`
    @param algorithm: the sklearn algorithm class used for clustering
    """

    for n in n_clusters:
        print(f"\n\nNumber of clusters: {n}")

        algorithm.set_params(n_clusters = n)
        labels = algorithm.fit(data.X).labels_

        results = []

        # For each n, print the highest risk clusters
        # To do that, analyze the single clusters
        for i in range(n):
            ith_cluster_patients = labels == i
            ith_patient_count = sum(ith_cluster_patients)
            # Risk mean calculation
            ith_cluster_diagnoses = np.delete(data.Y, np.where(labels != i))
            
            # To better understand what is a good cluster we need to introduce a figure of merit
            ith_risk = ith_cluster_diagnoses.mean()
            # Zero mean means zero risk! This is the only (reasonable) case in which the variance might be null
            ith_fom = (ith_patient_count * ith_risk) / (ith_cluster_diagnoses.var() * data.n_samples) if ith_risk > 0 else 0

            # Feature analysis
            feature_bounds = [
                (round(np.min(col), digits), round(np.max(col), digits), round(np.mean(col), digits))
                for col in filter_dataset(data.rescale(), lambda x: ith_cluster_patients[x[0]]).T
            ]
            
            results.append([i, round(ith_fom, digits)] + feature_bounds)
        
        # We can sort the array according to the score of our figure of merit
        results.sort(reverse=True, key=lambda cluster: cluster[1])

        fom_array = np.array([i[1] for i in results])
        print(f"Risk variance: %.2f" % (fom_array.var() * 100))
        print(f"Risk maxima: %.2f" % fom_array.max())

        # Generate DataFrame
        df = pd.DataFrame(results[:round(head*len(results))])
        yield df.rename(columns={old:new for old, new in enumerate(['cluster n°', 'risk fom'] + data.feature_names)})

def plot_inertia(X: np.array, n_max: int = None):
    """
    This function is used in the elbow method to 
    select the optimal number of clusters in KMeans.
    
    @param n_max: The maximum number of clusters. Defaults to n_samples
    """

    inertia = []
    end = X.shape[0] if not n_max else n_max
    for i in range(2, end):
        inertia.append(kmeans(X, i)[2])
    
    error_plot(inertia, range(2, end))