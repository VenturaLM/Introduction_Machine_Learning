import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import seaborn as sns
import sys


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

            corr = df.corr()
            sns.heatmap(corr, xticklabels=corr.columns,
                        yticklabels=corr.columns)
            plt.show()

    else:
        # Execution example:
        #   python3 .\exercise_8.py .\weather\weather.arff .\glass\glass.arff .\iris\iris.arff
        print("Data was not selected!")


if __name__ == "__main__":
    main()
