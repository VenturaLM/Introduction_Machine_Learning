import numpy as np
import scipy


dimension = int(input("Introduce la dimension de la matriz: "))

matriz = np.empty((dimension, dimension))

with np.nditer(matriz, op_flags=['readwrite']) as it:
   for x in it:
       x[...] = float(input("Introduce the next element: "))

print(matriz)

# MAXIMO POR FILAS Y POR COLUMNAS
print(matriz.max(axis=0)); # maximos por columnas
print(matriz.max(axis=1)); # maximos por filas

# DETERMINANTE DE LA MATRIZ
print(np.linalg.det(matriz))

# RANGO DE LA MATRIZ
print(np.linalg.matrix_rank(matriz))