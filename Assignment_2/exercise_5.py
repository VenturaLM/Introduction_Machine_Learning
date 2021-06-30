import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import seaborn as sns
import plotly.express as px


def main():
    data = arff.loadarff(".\iris\iris.arff")
    df = pd.DataFrame(data[0])

    sns.pairplot(data=df, hue='class')
    plt.show()

    pd.plotting.scatter_matrix(df)
    plt.show()

    df = px.data.iris()  # px dataframe.
    fig = px.scatter(df, x="sepal_width", y="sepal_length",
                     color="species", size='petal_length', hover_data=['petal_width'])
    fig.show()


if __name__ == "__main__":
    main()
