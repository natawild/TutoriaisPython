# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf 
import plotly.graph_objects as go 
from plotly import __version__
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('Classified Data', index_col=0)
print(df)

#standizire the data on the same scale 

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',  axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat)

#Agora os dados estão prontos para efetuarmos o o treino 

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


print('_______________k=1______________')
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

pred = knn.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

#Elbow method to choose a corret k value 

error_rate = []

for i in range(1,40):

	knn=KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train,y_train)
	pred_i = knn.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error rate vs k value')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()

print('________________k=17______________')
knnf = KNeighborsClassifier(n_neighbors=17)
knnf.fit(X_train,y_train)
predf= knnf.predict(X_test)

print(classification_report(y_test,predf))
print(confusion_matrix(y_test,predf))

