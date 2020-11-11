#linguagem de alto nivel, orientada a objetos, identação define blocos, nasceu em 1991
#desenvolve script -> chama o intrepretador python
# -*- coding:utf-8 -*- 
# import matplotlib.pyplot as plt 

print("Hello World")#mostra a mensagem no ecra
"""
comentário de muitas linhas

"""
mensagem= "Olá"
print (mensagem)

print(2+2)
print(2**3)#exponenciação
print(10/3)
print(10%3)#o resto (MÓDULO)


"""
OpERADORES LÓGICOS E RELACIONAIS
=--> Atribuição, x=10, x recebe o valor de 10

Operadores relacionais
== igual
!= difernte
> maior
< menor
>= maior ou igual
<= menor ou igual

Operadores Lógicos
AND --> DUAS CONDIÇÕES VERDADEIRA
OR --> PELO menos 1 condição seja verdadeira
NOT --> INVERT O VALOR

Comandos condicionais
if --> realiza testes condicionais, avalia se uma condição é verdadeira

if condição:
    executa esta linha

if condição:
    executa esta linha
else:
    caso o if falhe executa esta linha 


"""

x=2
y=3
print(x)
print (x==y)
print(x<y)

soma=x+y
print(soma==x)

print(x==y and x==soma)

a=1
b=100000000
c=2
print(a==b and b==c)
print(a==b or b==c)

#teste condicional 
if a > b:
    print("a é maior que b")

if a < b:
    print("b é maior que a")

#comando condicional else

t=1
u=2

if t > u:
    print("t maior que u")
else:
    print("t não é maior que u")

#comando elif

a=7
b=6

if b > a:
    if b > 0:
        print("b é maior que a\nb é positivo")
    else:
        print("b não é maior que a nem positivo")
else:
    print("b menor que a")  
    
    
a=-2 
b=-1   

if b > a:
    if b > 0:
        print("b é maior que a\nb é positivo")
    else:
        print("b não é maior que a nem positivo")
else:
    print("b menor que a")


x=1
y=2

#aparece a primeira que for verdadeira 
if x==y:
    print("numeros iguais")
elif x< y:
    print("x menor que y")
elif y> x:
    print("y maior que x")
else:
    print("numeors diferentes")


#laços de repetição

x=1
while x<10:
    print(x)
    x += 1 #x=x+1


lista = [1,2,3,4,5]
lista2 = ["ola", "mundo", "celia"]
lista3 = [0,1,"ola","bolacha", 9.99, True]


for i in lista:
    print(i)
    
    
for i in lista3:
    print(i)  

for i in range(10): 
    print(i)

for i in range(10,20,2): #imprime desde o 10 até 20 de 2 em dois 
    print(i)


a = "celia"
b = "luis"

print(a)
print(b)
concatenar = a + " " + b
print(concatenar)

tamanho=len(concatenar)
print(tamanho)

a = "Celia"
b = "luIs"
print(a)
print(a[2])

print(concatenar[0:5])
print(concatenar[1:])

#string = string.método()

print(concatenar.lower())
print(concatenar.upper())

#strip()- remove caracter especial
concatenar = a + " " + b + "\n"
print(concatenar)
print(concatenar.strip())

#split()

minhaString = "O rato roeu a roupa do rei de roma"

minhaLista = minhaString.split("r")
print(minhaLista)
busca = minhaString.find("rei")
print(busca)
print(minhaString[busca:])

#replace() substitui parte de string

minhaString = minhaString.replace("rei", "rainha")
print(minhaString)

#Funções palavra reservada def

def soma(x,y):
    return x+y 

s = soma(2,5)
print(s)

def multiplicacao (x,y): 
    return x*y

m = multiplicacao(3,4)
print(m)






