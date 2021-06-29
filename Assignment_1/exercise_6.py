"""
References:
	- Cramer's Rule:	https://rosettacode.org/wiki/Cramer%27s_rule#Python
"""

import numpy as np


def computeCramerRule(A, B, C):
    #	Result:
    X = []

    for i in range(0, len(B)):
        for j in range(0, len(B)):
            C[j][i] = B[j]
            if i > 0:
                C[j][i - 1] = A[j][i - 1]
        X.append(round(np.linalg.det(C) / np.linalg.det(A), 1))

    return X


def main():
    example = input(
        "\nMenu:\n1. 2x2 System equations.\n2. 3x3 System equations.\n3. 4x4 System equations.\n> ")

    if example == "1":
        #	Coefficients matrix:
        A = np.array([[4, -1], [3, 5]])
        #	Independent components:
        B = np.array([-9, -1])

    if example == "2":
        A = np.array([[3, 2, -1], [1, -1, 4], [5, -3, 1]])
        B = np.array([12, 19, 8])

    if example == "3":
        A = np.array([[2, -1, 5, 1], [3, 2, 2, -6],
                     [1, 3, 3, -1], [5, -2, -3, 3]])
        B = np.array([-3, -32, -47, 49])

    C = np.copy(A)

    X = computeCramerRule(A, B, C)
    print("\nSolution:", X)


if __name__ == "__main__":
    main()
