#Ficheiros

#Abrir: open r- só lê; w- escreve sempre num aquivo novo; a- leitura e escrita (adiciona o conteudo no fim do ficheiro)
#r+
#read()

ficheiro= open("ficheiro.txt")

linhas=ficheiro.readlines()
print(linhas)

textoCompleto = ficheiro.read()
print(textoCompleto)

w=open("ficheiro.txt", "w")

w.write("Este é o meu ficheiro lindo \n")

w.close()


