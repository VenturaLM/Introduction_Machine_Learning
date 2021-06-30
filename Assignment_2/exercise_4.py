import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.decomposition import PCA
import sys


plt.style.use("fivethirtyeight")


def loadFile(file_name):
    # Loads the data using dataframes.
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])

    return df


def main():
    if len(sys.argv) >= 2:
        for i in range(1, len(sys.argv)):
            file_name = sys.argv[i]
            df = loadFile(file_name)

            if file_name == ".\weather\weather.arff":
                df = df[['temperature', 'humidity']]

            if file_name == ".\iris\iris.arff":
                df = df[['sepallength', 'sepalwidth',
                         'petallength', 'petalwidth']]

            if file_name == ".\glass\glass.arff":
                df = df[['RI', 'Na', 'Mg', 'Al', 'Si', "'K'", 'Ca', 'Ba', 'Fe']]

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
            principalDf.hist()
            plt.show()

    else:
        # Execution example:
        #   python3 .\exercise_4.py .\weather\weather.arff .\glass\glass.arff .\iris\iris.arff
        print("Data was not selected!")


if __name__ == "__main__":
    main()
