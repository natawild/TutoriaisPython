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


ad_data = pd.read_csv('advertising.csv')
print(ad_data)

print(ad_data.info())
print(ad_data.describe())

#Create a histogram of the Age 

ad_data['Age'].plot.hist(bins=30)


#Create a jointplot showing 'Area Income' versus 'Age'

sns.jointplot(x='Age', y='Area Income', data=ad_data)

# Create a jointplot showing the kde distributions of Daily Time spent on site vs Age 

sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data, kind='kde', color='red')

#Create a jointplot of 'Daily Time Spent on Site' vs 'Daily Internet Usage'

sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color='green')


# Create a pairplot with the hue defined by the 'Clicked on Ad' column feature
sns.pairplot(ad_data,hue='Clicked on Ad')

#Split the data into training set and testing set using train_test_split
X = ad_data [['Daily Time Spent on Site','Age','Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#Train and fit a logistic regression model on the training set 

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#Predictions and Evaluations 
predictions = logmodel.predict(X_test)

#Create a classification report for the model 

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))



plt.show()




