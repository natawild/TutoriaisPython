#Escreva um programa que resolva uma equação de segundo grau.
import math 

a= int(input("Diga o valor de a:\n"))
b= int(input("Diga o valor de b:\n"))
c= int(input("Diga o valor de c:\n"))

delta = b**2 - (4*a*c)


if delta < 0:
    print("delta negativo:",delta)
    

else:
    raiz_delta = math.sqrt(delta)
    x1=(-b + raiz_delta)/2*a
    x2=(-b - raiz_delta)/2*a
    print("As raízes são", x1, "e", x2)
