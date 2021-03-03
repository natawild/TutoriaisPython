import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston 
from sklearn import metrics

customers = pd.read_csv('Ecommerce Customers')

dados= customers.head()
customers.info()
print('___________Data Explloration___________________')

#use seaborn to create a jointplot to compare the time on website ande yearly amount spent columns. Does the correlation make sense?

sns.jointplot(data=customers, x='Time on Website',y='Yearly Amount Spent')
#plt.show()


sns.jointplot(data=customers,x='Time on App', y='Yearly Amount Spent')
#plt.show()

# use a jointplot to create a 2D hexplot bin plot comparing Time on App and Length of Membership

sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)
#plt.show()

#Let's explore these types of relationship across the entire data set. Use pairplot to create the plot below.
sns.pairplot(customers)
#plt.show()

#Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs Length of Membership
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent', data=customers)
#plt.show()

print('______________Training and testing data___________')

#Set the variable X equal to the numerical features of the customers and variable y equal to the ' Yearly Amount Spent' column. 

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership', 'Yearly Amount Spent']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=101)

#Now its time to trains our model on our trainninf data

# create an instance of Linear Regression 
lm = LinearRegression()

#Train/fit lm on the training data 
lm.fit(X_train,y_train)

#print out the coefficientes of the model 
print(lm.coef_) 

#use lm.predict() to predict off the X_test set of the data 
predictions = lm.predict(X_test)

#create a scatter plot of the real test values versus the predited values. 
plt.scatter(y_test,predictions)
plt.xlabel('Y Test (True values)')
plt.ylabel ('Predicted Values')

print('__________Evaluating the Model________--')

print('MAE', metrics.mean_absolute_error(y_test,predictions))

print('MSE', metrics.mean_squared_error(y_test,predictions))

print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,predictions)))

print(metrics.explained_variance_score(y_test,predictions))

print('________________Residuals_________________')

#plot a histogram and make sure it 
sns.distplot(y_test-predictions,  bins= 50)
plt.show()

# coeffecient

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])

print(cdf)
