import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.decomposition import PCA


plt.style.use("fivethirtyeight")


def main():
    data = arff.loadarff(".\iris\iris.arff")
    df = pd.DataFrame(data[0])
    df = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
    print(df)
    print("Iris:\n", df.describe(), "\n")

    df.hist()
    plt.show()

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data=principalComponents, columns=[
        'Principal component 1', 'Principal component 2'])

    print(principalDf)
    print("Iris:\n", principalDf.describe(), "\n")

    principalDf.hist()
    plt.show()


if __name__ == "__main__":
    main()
