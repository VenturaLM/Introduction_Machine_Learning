"""	
References:
	- np.random.rand(rows, cols):		https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
	- np.amax(array) / np.amin(array):	https://numpy.org/doc/stable/reference/generated/numpy.amax.html
	- np.dot(array1, array2):			https://numpy.org/doc/stable/reference/generated/numpy.dot.html?highlight=dot#numpy.dot
"""

import numpy as np


def main():
    matrix_rows = int(input("\nSelect matrix rows:\n>"))
    matrix_cols = int(input("\nSelect matrix cols:\n>"))

    matrix = np.random.rand(matrix_rows, matrix_cols)
    print("\n", matrix)

    print("\nMaximum:", np.amax(matrix))
    print("Minimum:", np.amin(matrix))

    r1 = np.random.randint(0, matrix_rows)
    r2 = r1
    #	This "while" avoids using the same matrix row.
    while r2 == r1:
        r2 = np.random.randint(0, matrix_rows)

    c1 = np.random.randint(0, matrix_cols)
    c2 = c1
    #	This "while" avoids using the same matrix row.
    while c2 == c1:
        c2 = np.random.randint(0, matrix_cols)

    print("\nDot product - rows " + str(r1) + " and " +
          str(r2) + ":", np.dot(matrix[r1], matrix[r2]))
    print("Dot product - cols " + str(c1) + " and " +
          str(c2) + ":", np.dot(matrix[c1], matrix[c2]))

    print("\n")


if __name__ == "__main__":
    main()
