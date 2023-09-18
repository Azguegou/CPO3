# Récupérer le txt avec la matrice de poids
# Récupérer l'input liste de matrices
# Renvoyer liste de matrices

def save(matrices):
    save_file = open("data.txt", "w")
    for idMatrix in range(len(matrices)):
        matrix = matrices[idMatrix]
        for x in range(len(matrix)):
            line = matrix[x]
            for y in range(len(matrix[x])):
                save_file.write("%0.6f " % line[y])
            save_file.write("\n")
        save_file.write("\n\n")

    save_file.close()
