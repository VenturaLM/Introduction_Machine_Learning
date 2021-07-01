import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scipy.stats as stats


def decisionTrees(x_train, x_test, y_train, y_test, file_name):
    clf = DecisionTreeClassifier().fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print("\nDecision trees", file_name, " - Mean absolute error:",
          metrics.mean_absolute_error(y_test, y_predict))

    pd.DataFrame(y_test, y_predict).boxplot()
    plt.show()

    return y_predict


def SVM(x_train, x_test, y_train, y_test, file_name):
    clf = make_pipeline(StandardScaler(), SVC(
        gamma='auto')).fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print("SVM", file_name, " - Mean absolute error:",
          metrics.mean_absolute_error(y_test, y_predict))

    pd.DataFrame(y_test, y_predict).boxplot()
    plt.show()

    return y_predict


def k_nearest_neighbor(x_train, x_test, y_train, y_test, file_name):
    clf = KNeighborsClassifier(
        n_neighbors=6).fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print("K nearest neighbor", file_name, " - Mean absolute error:",
          metrics.mean_absolute_error(y_test, y_predict))

    pd.DataFrame(y_test, y_predict).boxplot()
    plt.show()

    return y_predict


def main():
    file_name = ".\datasets\cpu.arff"
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])

    x = df[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']]
    y = df['class']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)

    dt_predict = decisionTrees(x_train, x_test, y_train, y_test, file_name)
    svm_predict = SVM(x_train, x_test, y_train, y_test, file_name)
    knn_predict = k_nearest_neighbor(
        x_train, x_test, y_train, y_test, file_name)

    print()
    print("Decision Trees/SVM:", stats.wilcoxon(dt_predict, svm_predict))
    print("Decision Trees/KNN:", stats.wilcoxon(dt_predict, knn_predict))
    print("SVM/KNN:", stats.wilcoxon(svm_predict, knn_predict))
    print()
    print(stats.friedmanchisquare(dt_predict, svm_predict, knn_predict))
    print()


if __name__ == "__main__":
    main()
