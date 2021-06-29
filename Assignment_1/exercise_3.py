import numpy as np
import scipy


dimension = int(input("Select matrix size: "))

matriz = np.empty((dimension, dimension))

with np.nditer(matriz, op_flags=['readwrite']) as it:
    for x in it:
        x[...] = float(input("> "))

print("\n")
print(matriz)

# MAXIMO POR FILAS Y POR COLUMNAS
print("\n")
print(matriz.max(axis=0))  # maximos por columnas
print(matriz.max(axis=1))  # maximos por filas

# DETERMINANTE DE LA MATRIZ
print("\n")
print("Determinant:", np.linalg.det(matriz))

# RANGO DE LA MATRIZ
print("Range: ", np.linalg.matrix_rank(matriz))
