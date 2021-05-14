import numpy as np

'''
Ley D'Hont
Parámetros:
    - Número de Escaños
    - Número de Partidos
    - Votos por Partido
    - Para saber a quien le corresponde cada escanyo se calcula
      cada partido

      votosEscanyo = votosTotales / (escanyosYaConseguidos + 1)
'''

numeroEscanyos = int(input("Introduce el numero de escanyos: "))
entrada = 0

while(True):
    print("\nSelecciona el metodo de entrada de datos\n"
          "1. Fichero\n2. Consola")
    entrada = int(input("Opcion: "))

    if entrada in (1,2):
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
    numeroPartidos = int(input("Introduce el numero de partidos: "))
    votosPorPartido = np.ones((numeroPartidos,2))


    for partido in range(numeroPartidos):
        votos = int(input(f"Numero de votos para el partido {partido+1}: "))
        votosPorPartido[partido][0] = partido
        votosPorPartido[partido][1] = votos 

escanyosAsignados = np.ones((numeroPartidos, 1))

for escanyo in range(numeroEscanyos):
    partidoMasVotado = (0, 0) #(partido, numero de votos)

    for partido in range(len(votosPorPartido)):
        votosMetodoDHondt = votosPorPartido[partido][1] // escanyosAsignados[partido]

        if votosMetodoDHondt > partidoMasVotado[1]:
            partidoMasVotado = (partido, votosMetodoDHondt)
            

    escanyosAsignados[partidoMasVotado[0]] += 1

escanyosAsignados -= 1

for partido in range(len(escanyosAsignados)):
    print(f'El partido {partido+1} ha obtenido {int(escanyosAsignados[partido])} escanyos')