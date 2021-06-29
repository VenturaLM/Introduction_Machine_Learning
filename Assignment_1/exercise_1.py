import numpy as np

'''
Ley D'Hont
Parámetros:
    - Número de Escaños
    - Número de Partidos
    - Votos por Partido
    - Para saber a quien le corresponde cada escanyo se calcula
      cada partido

      votosEscanyo = votosTotales / (escañosYaConseguidos + 1)
'''

numeroEscaños = int(input("Select n seats: "))
entrada = 0

while(True):
    print("\nData browser\n"
          "1. File.\n2. Terminal.")
    entrada = int(input("> "))

    if entrada in (1, 2):
        break

if entrada == 1:
    with open("datosEj1.txt") as f:
        lineas = f.readlines()
        numeroPartidos = len(lineas)
        votosPorPartido = np.zeros((numeroPartidos, 2))

        for partido in range(len(lineas)):
            votosPorPartido[partido][0] = lineas[partido][0]
            votosPorPartido[partido][1] = lineas[partido][2:]

if entrada == 2:
    numeroPartidos = int(input("\nSelect n parties: "))
    votosPorPartido = np.ones((numeroPartidos, 2))

    for partido in range(numeroPartidos):
        votos = int(input(f"Party votes {partido+1}: "))
        votosPorPartido[partido][0] = partido
        votosPorPartido[partido][1] = votos

escañosAsignados = np.ones((numeroPartidos, 1))

for escanyo in range(numeroEscaños):
    partidoMasVotado = (0, 0)  # (partido, numero de votos)

    for partido in range(len(votosPorPartido)):
        votosMetodoDHondt = votosPorPartido[partido][1] // escañosAsignados[partido]

        if votosMetodoDHondt > partidoMasVotado[1]:
            partidoMasVotado = (partido, votosMetodoDHondt)

    escañosAsignados[partidoMasVotado[0]] += 1

escañosAsignados -= 1

print("\n")
for partido in range(len(escañosAsignados)):
    print(
        f'Party {partido+1} has obtained {int(escañosAsignados[partido])} seats.')

print("\n")
