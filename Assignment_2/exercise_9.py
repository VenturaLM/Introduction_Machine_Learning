import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
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

            if file_name == ".\weather\weather.arff":
                aux_df = df[['temperature', 'humidity', 'play']]
                pd.plotting.parallel_coordinates(aux_df, 'play')

            if file_name == ".\iris\iris.arff":
                pd.plotting.parallel_coordinates(df, 'class')

            if file_name == ".\glass\glass.arff":
                pd.plotting.parallel_coordinates(df, 'Type')

            plt.show()

    else:
        # Execution example:
        #   python3 .\exercise_9.py .\weather\weather.arff .\glass\glass.arff .\iris\iris.arff
        print("Data was not selected!")


if __name__ == "__main__":
    main()
