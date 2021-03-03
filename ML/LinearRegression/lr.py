import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston 
from sklearn import metrics 
#%matplotlib inline 

df = pd.read_csv('USA_Housing.csv')

# split data into train and test 

df.columns 

X = df[['Avg. Area Income','Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=101)

lm = LinearRegression()
 
lm.fit(X_train,y_train)

print(lm.intercept_)

lm.coef_ 

X_train.columns

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])

print(cdf) 


print('--------- Boston Data ------------- ')

boston = load_boston()
chaves = boston.keys()
print(chaves)
print(boston['DESCR'])


print('--------- Predictions ------------- ')

predictions = lm.predict(X_test)
print(predictions)

sc = plt.scatter(y_test,predictions)
plt.show()

#residuals 
sns.displot((y_test-predictions))
plt.show()

metrics.mean_absolute_error(y_test,predictions)

metrics.mean_squared_error(y_test,predictions)

raiz = np.sqrt(metrics.mean_squared_error(y_test,predictions))
print(raiz)


