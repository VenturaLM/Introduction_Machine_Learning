import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import seaborn as sns
import sys
from sklearn.decomposition import PCA


def loadFile(file_name):
    # Loads the data using dataframes.
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])

    return df


def normalizeData(df):
    df.iloc[:, 0:4] = df.iloc[:, 0:4].apply(
        lambda x: (x - x.min()) / x.max() - x.min(), axis=0)

    sns.pairplot(data=df, hue='class')
    plt.show()


def standardizeData(df):
    # Standardizes dataset.
    df = (df - df.mean()) / df.std()

    return df


def main():
    if len(sys.argv) >= 2:
        for i in range(1, len(sys.argv)):
            file_name = sys.argv[i]
            df = loadFile(file_name)

            if file_name == ".\weather\weather.arff":
                aux_df = df[['temperature', 'humidity']]

            if file_name == ".\iris\iris.arff":
                aux_df = df[['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth']]

            if file_name == ".\glass\glass.arff":
                aux_df = df[['RI', 'Na', 'Mg', 'Al',
                             'Si', "'K'", 'Ca', 'Ba', 'Fe']]

            df = standardizeData(aux_df)
            print(df)
            print("\n", df.describe(), "\n")

            df.hist()
            plt.show()

            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(df)
            principalDf = pd.DataFrame(data=principalComponents, columns=[
                'P1', 'P2'])

            print(principalDf)
            print("\n", principalDf.describe(), "\n")

            pd.plotting.scatter_matrix(principalDf)
            plt.show()

    else:
        # Execution example:
        #   python3 .\exercise_7.py .\weather\weather.arff .\glass\glass.arff .\iris\iris.arff
        print("Data was not selected!")


if __name__ == "__main__":
    main()
