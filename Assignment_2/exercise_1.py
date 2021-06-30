import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import arff
# from sklearn.preprocessing import StandardScaler

# References:
#   https://www.python-course.eu/dealing_with_NaN_in_python.php
#   https://datatofish.com/import-csv-file-python-using-pandas/
#   https://www.geeksforgeeks.org/command-line-arguments-in-python/
#   Histograms: https://moonbooks.org/Articles/How-to-create-and-plot-a-simple-histogram-with-matplotlib-and-python-/
#               https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

# Videos:
#   https://www.youtube.com/watch?v=XDv6T4a0RNc

plt.style.use("fivethirtyeight")


def loadFile(file_name):
    # Loads the data using dataframes.
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])

    return df


def normalizeData(file_name, df):
    # Normalizes dataset.
    df = (df - df.min()) / (df.max() - df.min())
    printNormalizeData(file_name, df)


def printNormalizedData(file_name, df):
    # Prints normalization data.
    print("\n", file_name, " [0, 1]:\n\n", df)
    df.hist()
    plt.show()


def standardizeData(file_name, df):
    # Normalizes dataset.
    df = (df - df.mean()) / df.std()
    printStandardizedData(file_name, df)


def printStandardizedData(file_name, df):
    # Prints normalization data.
    print("\n\n", file_name,
          "[μ = 0, σ = 1] statistics:\n\n", df.describe(), "\n")
    df.hist()
    plt.show()


def main():
    if len(sys.argv) >= 2:
        for i in range(1, len(sys.argv)):
            file_name = sys.argv[i]
            df = loadFile(file_name)

            # Computes dataframe histogram.
            df.hist(alpha=0.7)
            plt.show()

            if file_name == ".\weather\weather.arff":
                df = df[['temperature', 'humidity']]

            if file_name == ".\iris\iris.arff":
                df = df[['sepallength', 'sepalwidth',
                         'petallength', 'petalwidth']]

            if file_name == ".\glass\glass.arff":
                df = df[['RI', 'Na', 'Mg', 'Al', 'Si', "'K'", 'Ca', 'Ba', 'Fe']]

            # Normalization.
            normalizeData(file_name, df)

            # Standardization.
            standardizeData(file_name, df)
    else:
        print("Data was not selected!")


if __name__ == "__main__":
    main()
