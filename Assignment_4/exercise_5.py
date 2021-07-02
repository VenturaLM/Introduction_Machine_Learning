import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.io import arff


def main():
    file_name = ".\datasets\glass2.arff"
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])

    x = df[['RI', 'Na', 'Mg', 'Al', 'Si', "'K'", 'Ca', 'Ba', 'Fe', 'Type']]
    y = df["Type"]

    n_classes = 7

    for i in range(n_classes):
        kmeans = KMeans(n_clusters=i, random_state=1).fit(x)
        ss = metrics.silhouette_score(x, kmeans.labels_, metric='euclidean')
        print("\t Kmeans", i, "n_cluster = ", ss)


if __name__ == "__main__":
    main()
