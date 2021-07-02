"""
    References: https://www.youtube.com/watch?v=v7oLMvcxgFY
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import random
from sklearn import datasets

from sklearn.cluster import AgglomerativeClustering
from scipy.io import arff


def main():
    file_name = ".\datasets\glass.arff"
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])

    headers = df.columns
    column_1 = headers[random.randint(0, len(headers))]
    column_2 = headers[random.randint(0, len(headers))]
    #['RI', 'Na', 'Mg', 'Al', 'Si', "'K'", 'Ca', 'Ba', 'Fe']

    dendogram = sch.dendrogram(sch.linkage(df, method="ward"))

    df.plot.scatter(x=column_1, y=column_2)
    plt.show()


if __name__ == "__main__":
    main()
