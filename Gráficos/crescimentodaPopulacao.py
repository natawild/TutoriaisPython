#crescimento da populacao 
import matplotlib.pyplot as plt 

dados = open("populacao_brasileira.csv").readlines()

x=[]
y=[]

for i in range(len(dados)): #range cria uma lista do tamanho dos dados 
	if i!= 0: #ignorar a primeira linha 
		linha = dados[i].split(";")
		x.append(int(linha[0]))
		y.append(int(linha[1]))

print(y)
plt.plot(x,y, color="k", linestyle="--")
plt.bar(x,y, color="#e4e4e4")
#plt.scatter(x,y, color="k")
plt.title("Crescimento da populacao brasileira 1980-2016")
plt.xlabel("Ano")
plt.ylabel("Populacão x 100000")
plt.show()
plt.savefig("grafPopbrasileita.png", dpi=300) #para guardar a imagem


