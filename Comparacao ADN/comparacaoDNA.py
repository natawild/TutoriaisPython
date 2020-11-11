#comparacao de RNA ribossomal 

entrada = open("human.fasta").read()
saida = open("humano.html","w")

contagem = {}


for i in ["A","T","C","G"]:
	for j in ["A","T","C","G"]:
		contagem[i+j] = 0
print(contagem)

entrada = entrada.replace("\n","")

for k in range(len(entrada)-1): #tamanho da entrada 
	contagem[entrada[k]+entrada[k+1]] +=1 #soma 1 quanto encontra a sequencia 

print(contagem)


#html 
saida.write("<div>")

i = 1 
for k in contagem: 
	transparencia = contagem[k]/max(contagem.values())
	saida.write("<div style='width:100px; border:1px solid #111; color:#fff; height:100px; float:left; background-color:rgba(0, 0, 0, "+str(transparencia)+"')>"+k+"</div>")
	if i%4 == 0: 
		saida.write("<div style='clear:both'></div>")
	i+=1

saida.close()