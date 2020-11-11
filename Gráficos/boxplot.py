#boxplot ou diagrama de caixa 

import matplotlib.pyplot as plt 
import random 

vetor = [1,2,3,4,5,67,8,909,34]

plt.boxplot(vetor)
plt.show()

#numeros aleatóreos 
v = []

for i in range(100):
	nrAleatorio= random.randint(0,50)
	v.append(nrAleatorio)

plt.boxplot(v)
plt.title("Boxplot")
plt.show()

