import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import seaborn as sns
import plotly.express as px
import sys


def normalizeData(df):
    df.iloc[:, 0:4] = df.iloc[:, 0:4].apply(
        lambda x: (x - x.min()) / x.max() - x.min(), axis=0)

    sns.pairplot(data=df, hue='class')
    plt.show()


def standardizeData(df):
    df.iloc[:, 0:4] = df.iloc[:, 0:4].apply(
        lambda x: (x - x.mean()) / x.std(), axis=0)

    sns.pairplot(data=df, hue='class')
    plt.show()


def main():
    data = arff.loadarff(".\iris\iris.arff")
    df = pd.DataFrame(data[0])

    sns.pairplot(data=df, hue='class')
    plt.show()

    normalizeData(df)

    standardizeData(df)


if __name__ == "__main__":
    main()
