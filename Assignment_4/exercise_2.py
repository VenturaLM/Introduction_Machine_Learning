"""
    References: https://www.youtube.com/watch?v=s6PSSzeUMFk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn import cluster

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def normalizeData(df):
    # Normalizes dataset.
    df = (df - df.min()) / (df.max() - df.min())
    return df


def computeElbowMethod(df):
    # Look for the optimun number of clusters.
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, max_iter=300)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)

    plotWCSSResults(wcss)


def plotWCSSResults(wcss):
    plt.plot(range(1, 11), wcss)
    plt.title("Elbow Method")
    plt.xlabel("n Clusters")
    plt.ylabel("WCSS")
    plt.show()


def computePCA(df, normed_df):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(normed_df)
    pca_data_df = pd.DataFrame(data=pca_data, columns=[
                               "Component_1", "Component_2"])
    pca_headers_df = pd.concat([pca_data_df, df[["KMeans_Clusters"]]], axis=1)

    return pca_headers_df


def plotResults(pca_headers_df):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Component 1", fontsize=15)
    ax.set_ylabel("Component 2", fontsize=15)
    ax.set_title("PCA", fontsize=20)

    color_theme = np.array(["blue", "green", "orange"])
    ax.scatter(x=pca_headers_df.Component_1, y=pca_headers_df.Component_2,
               c=color_theme[pca_headers_df.KMeans_Clusters], s=50)
    plt.show()


def main():
    file_name = ".\datasets\glass.arff"
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])

    # Normalize data.
    normed_df = normalizeData(df)
    print(normed_df)
    print()
    print(normed_df.describe())

    # Look for the optimun number of clusters: Elbow Method.
    computeElbowMethod(normed_df)

    # Apply K-Means method to dataset.
    clustering = KMeans(n_clusters=2, max_iter=300)
    clustering.fit(normed_df)

    # Adds the classificaton to the original dataset.
    df["KMeans_Clusters"] = clustering.labels_
    print()
    print(df.head())

    # PCA.
    pca_headers_df = computePCA(df, normed_df)

    # Results plotting.
    plotResults(pca_headers_df)


if __name__ == "__main__":
    main()
