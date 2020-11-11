#list comprehension

x=[1,2,3,4,5]
y=[]

for i in x:
    y.append(i**2)

print(x)
print(y)

#lista=[valor_a_adicionar laço condiçao]


z=[i**2 for i in x]
print(x)
print("list comprehension: ", z)

w=[i for i in x if i%2==1]
print("Números ímpares:", w)

#funçao enumerate 

lista = ["Abacate", "leite","ananas"]

for i in range(len(lista)):
    print(i, lista[i])


for i, nome in enumerate(lista): #ESTA FAZ O MEMSO QUE EM CIMA 
    print(i,nome)


#map

def dobro(x):
    return x*2

valor = 2
print(dobro(valor))


valor2 = [1,2,3,4,5]
print(dobro(valor2))

print(map(dobro,valor2)) #assim não mostra o valor, apenas diz onde ele está guardado

valorDobrado = map(dobro, valor2)
for v in valorDobrado: 
      print(v)

valorDobradoLista = map(dobro, valor2)
valorDobradoLista = list(valorDobradoLista)
print(valorDobradoLista) 


#reduce, recebe uma lista e retorna a penas um valor

from functools import reduce

def soma(x, y):
    return x+y

listaRedduce = [1, 3, 5, 7, 9, 13]
soma = reduce(soma, listaRedduce)
print(soma)

#zip

l1 = [1,2,3,4,5]
l2 = ["abacate","bola","cebola","batatta","ola"]
l3 = ["5€","4€","5€","4€","1€"]

for nr, nome, valor in zip(l1,l2,l3):
    print(nr,nome,valor)


lista = [1,2,3,4,5,6,7,8,9,10]
lista2 = []

def dob(i):
    if i%2==0:
        return i
#lista2 = [i%2 for i in lista]
lista2 = filter(dob,lista)

print(lista2)


print(map(dob,lista2)) #assim não mostra o valor, apenas diz onde ele está guardado

valorDobrado = map(dob, lista2)
for v in valorDobrado: 
      print(v)



x = [1, 2, 3]
y = [i for i in x if i % 2 == 0]
 
print(y)
