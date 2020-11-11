nome= "Célia Figueiredo"
print(nome)


#string, tem posições que começa em 0

nome[0:5]
nome[1:5]

nome+nome

"a" in nome 

#Listas, conjunto de variáveis numéricas ou strings numeradas por uma certa ordem

lista=[1,4,7,"celia",23,14]
print(lista)
lista.append("luis")
lista.append(20)
print(lista)

lista.index("luis")

lista.count(4)
lista.append(4)
lista.count(4)
lista.remove(4)
lista.reverse()

lista2=[1,9,3,4,1,7,6]
lista2.sort()


#dicionários

telefones={"celia":939393939,"luis":919191919,"carla":96969669}
print(telefones)
#adicionar 
telefones["rita"]=94949494949
#retirar 
del telefones["carla"]


#tuplas

tuplas=("celia","luis","carla")
tuplas


tuplas[0]
tuplas[1]
tuplas[2]

tuplas[0:2]

len(tuplas)
tuplas+tuplas
tuplas*5

4 in tuplas
"celia" in tuplas

listaNova=[1,2,4,"celia"]

tuplas2=tuple(listaNova)
tuplas2

#if
numero=1
if(numero==1) :
    print("Nr é igual a 1")

if(numero==2):
    print("Nr é igual a 2")


if(numero==1):
    print("Numero é igual a 1)"
else:
    print("Numero não é igual a 1")


if(numero==1):
	print("sim")
else:
	print("não")

nome="celia"
if("z" in nome):
        print("z está no nome ")

elif ("c" in nome):
        print("nome tem a letra c")

else:
        pass


#For loop
for x in range(0,5):
        print("Valor de x é:", x)


nome="celialinda"
for letra in nome:
    print(letra)


lista=["celia",19,"portugal"]
for valor in lista:
    print(valor)


#while loop
numero=15
while(numero>0):
    print(numero)
    numero=numero-1
    
#uso de break (interrompe um ciclo) and continue (ignora esse numero, avança pasa o seguinte)

numero=20
while True:
    numero=numero-1
    print(numero)
    if(numero==2):
        break

numero = 10
while True:
    numero=numero-1
    if(numero==4):
        continue
    print(numero)
    if(numero==2):
        break

#pass não faz nada (usado quando ainda não sabemos o que por num ciclo)

for x in range(0,5):
    pass 
    





    




    
    



    
    
    
    



