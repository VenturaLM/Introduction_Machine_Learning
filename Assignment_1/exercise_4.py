"""	
References:
	- np.mean(array) / np.median(array) / np.mode(array): 	https://www.w3schools.com/python/python_ml_mean_median_mode.asp
"""

import numpy as np
from scipy import stats as st


def main():
    matrix_rows = int(input("Select matrix rows:\n>"))
    matrix_cols = int(input("Select matrix cols:\n>"))

    matrix = []

    print("\n")
    for i in range(matrix_rows):
        row = []
        for j in range(matrix_cols):
            value = int(input("> "))
            row.append(value)
        matrix.append(row)

    print("\n")
    print(np.matrix(matrix))

    print("\nMode: ", st.mode(np.matrix(matrix)).mode)
    print("Mean: ", np.mean(np.matrix(matrix)))


if __name__ == "__main__":
    main()
