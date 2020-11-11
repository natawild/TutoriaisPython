#Listas

minhaLista = [ "morango", "melancia", "abacate"]
minhaLista2 = [1,2,3,4,5,6]
minhaLista3 = ["abacaxi", 2, 9.99, True]
minhaLista4 = []
print(minhaLista)
print(minhaLista2) 

print(minhaLista2[2]) 

tamanho = len(minhaLista)
print(tamanho)

for i in minhaLista:
    print(i)

#adicionar elementos

minhaLista.append("limão")
print(minhaLista)

#verificar se um elemento existe na lista

if 7 in minhaLista2:
    print("7 está na lista")
    
del minhaLista2[2:]
print(minhaLista2)

minhaLista4.append(57)
print(minhaLista4)


listax = [200,3304040,44,55,6,6,70780,80,0]
listax.sort()
print(listax)

listan = ["bola", "abacate","dinheiro"]

listan.sort()
print(listan)



#Dicionários
# dicionário={'chave:'valor'}

meuDicionario={"a":"Ameixa","b":"Bola","c":"cadela"}
print(meuDicionario)

for chave in meuDicionario:
    print(chave+"_"+meuDicionario[chave])

for i in meuDicionario.items():
    print(i)

for i in meuDicionario.values():
    print(i)

for i in meuDicionario.keys():
    print(i)

#numeros aleatórios 

import random
random.seed(1)#assim escolhe sempre o mesmpo numero 
numero= random.randint(0,10)
print(numero)



lista = [6, 45,9]
nr = random.choice(lista)
print(nr)



#Excecões

a=2
b=0
try: 
    print(a/b)
except:
    print("Não pode dividir por 0")

print(a/a)
