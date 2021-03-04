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


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from IPython.display import Image 

url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300,height=300)

iris = sns.load_dataset('iris')

print('_____________________Exploratory Data Analysis_____________________________')

#create a pairplot of the data set. Wich flower species seems to be the most separable? 
sns.pairplot(iris,hue='species',palette='Dark2')
plt.show()

setosa = iris[iris['species']=='setosa']

sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True,shade_lowest=False)
plt.show()

print('_____________________Train Test Split_____________________________')

X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

print('_____________________Train Model_____________________________')

svc_model = SVC()
svc_model.fit(X_train,y_train)

print('_____________________Model Evaluation_____________________________')

predictions = svc_model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


print('_____________________Gridsearch Practice_____________________________')
#create a dictionary called param_grid and fill out some parametres for C and gamma

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


#create a GridSearchCV object and fit it to the training data
grid = GridSearchCV(SVC(),param_grid,verbose=2)

grid.fit(X_train,y_train)
grid.best_params_ 
grid.best_estimator_ 
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))




