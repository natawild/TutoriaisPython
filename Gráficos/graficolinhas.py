#gráfico de linhas

import matplotlib.pyplot as plt 

x = [1,2,5]
y = [2,3,7]

#títulos 
plt.title("O Título do grágico")

#Eixos 
plt.xlabel("Eixo dos xx ")
plt.ylabel("Eixo dos yy ")

plt.plot(x,y) #desenha o gráfico 
plt.show() #mostra o gráfico 

