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


cancer = load_breast_cancer()
print(cancer.keys())

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df_feat)

print(cancer['target_names'])

print('_____________________Train Test Split_____________________________')


X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


print('_____________________Training Model_____________________________')
model = SVC()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(),param_grid,verbose=3)

grid.fit(X_train,y_train)
grid.best_params_ 
grid.best_estimator_ 
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))









