
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


df = pd.read_csv('KNN_Project_Data', index_col=0)
print(df)

#use seaborn on the dataframe to cretae a pairpllot with the hue indicated by the TARGET CLASS column 
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


#Create a StandardScaler() object called scaler
scaler = StandardScaler() 

#Fit scaler to the features 
scaler.fit(df.drop('TARGET CLASS',  axis=1))

#use the transform() method to transform the features to a scaled version 
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

#Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked 
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat)

print('_____________________Train Test Split_____________________________')
#use train_test_split to split data into a training set and testing set 

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

print('_____________________Using knn_____________________________')
#import KNeighborsClassifier from skikit learn 
#create a knn model instance with n_neighbors=1
knn = KNeighborsClassifier(n_neighbors=1)

#fit this knn model to the training data 
knn.fit(X_train,y_train)

print('_____________________Predictions and Evaluations_____________________________')
#use the predict method to predict values using your KNN model and X_test. 
pred = knn.predict(X_test)

#create a confusion matrix and classification report
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

print('_____________________Choosing a K value_____________________________')

#create a loop that trains various KNN models with different k vales, then keep track of the error_rate for each of these models with a list.
error_rate = []

for i in range(1,60):

	knn=KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train,y_train)
	pred_i = knn.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test))


#now create to following plolt using the information from your loop
plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error rate vs k value')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()

print('_____________________Retrain with new k value (30)_____________________________')
#retrain your model with the best K value and re-do the classification report and the confusion matrix
knnf = KNeighborsClassifier(n_neighbors=30)
knnf.fit(X_train,y_train)
predf= knnf.predict(X_test)

print(classification_report(y_test,predf))
print(confusion_matrix(y_test,predf))


print('_____________________Retrain with new k value (37)_____________________________')
#retrain your model with the best K value and re-do the classification report and the confusion matrix
knnf = KNeighborsClassifier(n_neighbors=37)
knnf.fit(X_train,y_train)
predf= knnf.predict(X_test)

print(classification_report(y_test,predf))
print(confusion_matrix(y_test,predf))


