# Récupérer le txt avec la matrice de poids
# Récupérer l'input liste de matrices
# Renvoyer liste de matrices
import numpy as np


def save(matrices):
    save_file = open("data.txt", "w")
    for idMatrix in range(len(matrices)):
        matrix = matrices[idMatrix]
        for x in range(len(matrix)):
            line = matrix[x]
            for y in range(len(matrix[x])):
                save_file.write("%0.6f " % line[y])
            save_file.write("\n")
        save_file.write("END_MATRIX")
    save_file.close()


def load():
    matrixList = []
    for matrix in open("data.txt", "r").read().split("END_MATRIX")[0:-1]:
        matrixWIP = []
        for line in matrix.split("\n")[0:-1]:
            lineWIP = [float(value) for value in line.split(" ")[0:-1]]
            matrixWIP.append(lineWIP)
        matrixList.append(matrixWIP)
    return matrixList
