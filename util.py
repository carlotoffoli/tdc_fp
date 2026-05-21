import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from pandas import Series

def accuracy(prediction: np.array, ground_truth: np.array):
    """
    This calculates the accuracy against the ground truth
    Usable also in multiclass classification
    """
    if prediction.shape[0] != ground_truth.shape[0]:
        raise Exception(f"Different sizes detected: {prediction.shape[0]} and {ground_truth.shape[0]}")
    return np.sum(prediction == ground_truth) / len(prediction)

def pca_decomposition(X: np.array):
    pca = PCA(n_components = 3).fit(X)
    return pca.transform(X), pca.components_

def plot_clusters(X_pca, labels, title):
    """
    This is used to plot PCA-reduced data (3D).

    @param X_pca: The PCA-reduced data (3D)
    @param label: The cluster labels
    """
    # From sklearn PCA example on iris dataset

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=100)
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=Series(labels.ravel()),
        s=40,
    )

    ax.set(
        title=title,
        xlabel="1st Principal Component",
        ylabel="2nd Principal Component",
        zlabel="3rd Principal Component",
    )
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    # Add a legend
    legend1 = ax.legend(
        scatter.legend_elements()[0],
        [0, 1],
        loc="upper right",
        title=title,
    )
    ax.add_artist(legend1)

    plt.show()