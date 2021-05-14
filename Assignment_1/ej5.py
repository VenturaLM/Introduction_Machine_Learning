import numpy as np

# LEER MATRIZ DE FICHERO
try:
    matriz = np.loadtxt("datosEj5.txt")
except IOError:
    print("El fichero no exite!")
    quit()

np.set_printoptions(precision=3, suppress=True)

# INVERSA DE MATRIZ
inv = np.linalg.inv(matriz)

print(inv)

# PRODUCTO DE MATRICES
print(np.dot(inv, matriz))
