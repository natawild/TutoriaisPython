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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


loans = pd.read_csv('loan_data.csv')
print(loans)

loans.info()


print('_____________________Exploratory Data Analysis_____________________________')
#Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='Credit Policy = 1')
loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label='Credit Policy = 0')
plt.legend()


#Create a similar figure, exect this time select by the not.fully.paid column 
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,color='blue',label='Not Fully paid = 1')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,color='red',label='Not Fully paid = 0')
plt.legend()


#create a countplot using seaborn showing the counts of loans by purpose, withe the color hue defined by not.fully.paid
plt.figure(figsize=(10,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


#let's see the trend betweeen fico score and interest rate. Recreate the following jointplot
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


#create lmplots to see if the trend differed between not.fully.paid and credir.policy. 
#Check the documentation for lmpllot() if you can't figure out how to separte it into columns 
plt.figure(figsize=(10,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')
plt.show()

print('_____________________Categorical Features_____________________________')
#create a list of 1 element containing the string 'purpose'

cat_feats = ['purpose']

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(final_data)


print('_____________________Train Test Split_____________________________')
#use sklearn to split data into a training set and a testing set 

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

print('_____________________Training a Decision Tree Model_____________________________')

#TREINAR APENAS 1 ÁRVORE DE DECISÃO 

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


print('_____________________Predictions and Evaluate of Decision Tree_____________________________')
#create predictions from the test set and create a classifocation report and a confusion matrix 

predictions = dtree.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


print('_____________________Training the Random Forest model _____________________________')
#create an instance of the RandomForestClassifier class and fit it to our training data from the previous step 

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)

print('_____________________Predictions and Evaluation____________________________')
#predict the class of not.fully.paid for the X_test data

rfc_pred = rfc.predict(X_test)

#show the confusion matrix for the predictions
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


