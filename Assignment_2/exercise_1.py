import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler

# References:
#   https://www.python-course.eu/dealing_with_NaN_in_python.php
#   https://datatofish.com/import-csv-file-python-using-pandas/
#   https://www.geeksforgeeks.org/command-line-arguments-in-python/
#   Histograms: https://moonbooks.org/Articles/How-to-create-and-plot-a-simple-histogram-with-matplotlib-and-python-/
#               https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

# Videos:
#   https://www.youtube.com/watch?v=XDv6T4a0RNc

plt.style.use("fivethirtyeight")


def histogram(data):
    # Computes histogram from an array.
    plt.subplot(211)
    plt.hist(data, edgecolor="black")
    plt.title("Histogram", fontsize=10)


def normalizedHistogram(data):
    # Computes normalized histogram from an array.
    plt.subplot(212)
    plt.hist(data, density=True, edgecolor="black")
    plt.title("Normalized histogram", fontsize=10)


def standardizedHistogram(data):
    # TODO
    # Computes standardized histogram from an array.
    # Note: https://machinelearningmastery.com/normalize-standardize-time-series-data-python/
    #       https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    x = data.loc[:, data].values
    y = data.loc[:, ['target']].values

    x = StandardScaler().fit_transform(x)


def loadFile(file_name):
    data = pd.read_csv(file_name, sep=";")
    df = pd.DataFrame(data)

    # histogram(df['age'])
    # normalizedHistogram(df['age'])
    standardizedHistogram(df)

    # plt.show()


def main():
    if len(sys.argv) == 2:
        loadFile(sys.argv[1])


if __name__ == "__main__":
    main()
