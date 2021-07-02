"""
    References:
        https://www.youtube.com/watch?v=v7oLMvcxgFY
        https://www.youtube.com/watch?v=iT4xYghI7Rg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch

from scipy.io import arff
from sklearn.cluster import SpectralClustering


def main():
    file_name = ".\datasets\glass.arff"
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])

    # ['RI', 'Na', 'Mg', 'Al', 'Si', "'K'", 'Ca', 'Ba', 'Fe']
    attributes = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].values

    h_clustering = sch.linkage(attributes, method="ward")
    dendogram = sch.dendrogram(h_clustering)

    # El parámetro t indica dónde corto en el dendograma, en el eje de la y.
    clusters = sch.fcluster(h_clustering, t=15, criterion="distance")
    n_clusters = np.amax(np.unique(clusters))

    df["Hierarchical Clustering"] = clusters
    # print(df)

    # Introducir en el eje x e y alguna de las columnas para ver el nivel de relación entre las dos variables.
    df.plot.scatter(x="RI", y="Si", c=clusters,
                    cmap="Greens", edgecolor="black")

    plt.show()


if __name__ == "__main__":
    main()
