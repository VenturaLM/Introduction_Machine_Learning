import numpy as np

# LEER MATRIZ DE FICHERO
try:
    matriz = np.loadtxt("matrix.txt")
except IOError:
    print("File does not exists.")
    quit()

np.set_printoptions(precision=3, suppress=True)

# INVERSA DE MATRIZ
inv = np.linalg.inv(matriz)

print("\n")
print("Inverse: ", inv)

# PRODUCTO DE MATRICES
print("\n")
print("Product: ", np.dot(inv, matriz))
