import matplotlib.pyplot as plt 

x = [1,2,5,6,7]
y = [2,3,7,1,0]

titulo = "Gráfico de barras"
eixox = "Eixo do x "
eixoy = "Eixo do y"

plt.title(titulo)
plt.xlabel(eixox)
plt.ylabel(eixoy)

plt.bar(x,y)#desenha o gráfico 
plt.show() #mostra o gráfico 

#

x1 = [1,3,5,7,9]
y1 = [2,3,7,1,0]

x2 = [2,4,6,8,10]
y2 = [5,1,3,7,4]


titulog2 = "Gráfico de barras"
eixoxg2 = "Eixo do x "
eixoyg2 = "Eixo do y"
#legendas 

plt.title(titulog2)
plt.xlabel(eixoxg2)
plt.ylabel(eixoyg2)

plt.bar(x1,y1, label = "Grupo 1")#desenha o gráfico 
plt.bar(x2,y2, label = "Grupo 2")
plt.legend()
plt.show()