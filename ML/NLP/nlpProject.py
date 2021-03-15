import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import nltk 
import string 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report, confusion_matrix
sns.set_style('white')


yelp = pd.read_csv('yelp.csv')

print(yelp.head())

print(yelp.info())

print(yelp.describe())

#acrescentar uma nova coluna com o nome text length, calculando o tamanho da coluna text
yelp['text length'] = yelp['text'].apply(len)
print(yelp)

print('________________Evaluation Data Analisys__________________')

#Use FacetGrid to cretae a grid of 5 histograms of text length based off of the satr ratings. 

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length',bins=50)
plt.show()

#create a boxplot of text length for each stra category 
sns.boxplot(x='stars',y='text length', data=yelp, palette='rainbow')
plt.show()

#create a countplot of the number of occurrences for each type of star rating 
sns.countplot(x='stars',data=yelp,palette='rainbow')
plt.show()

#use groupby to get the mean values of the numeriacal columns. 
stars = yelp.groupby('stars').mean()
print(stars)

#use the corr() method on that groupby dataframe to produce this dataframe 
print(stars.corr())

#then use seaborn to create a heatmap based off that .corr() dataframe
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)
plt.show()

print('________________NLP Classification Task__________________')

#create a dataframe called yelp_class that contains the column of yelp dataframe but for only the 1 or 5 star reviews 
yelp_class = yelp[(yelp['stars']==1)|(yelp['stars']==5)]


#create two objects X and y. Will be the 'text' column of yelp_class and y will be thhe 'stars' column of yelp_class 
X = yelp_class['text']
y = yelp_class['stars']

cv = CountVectorizer()

#USE THE FIT_TRANSFORM method on the CounterVectorizer object and pass in X. 
X = cv.fit_transform(X)


print('________________Train Test Split__________________')
#use train_test_split to split up the data into X_train, X_test, y_train, y_test; use test_size=0.3 random_state=101

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)


print('________________Training a Model__________________')

#create an instance of the estimator and all is nb 
nb = MultinomialNB()

#fit nb using training data 
nb.fit(X_train,y_train)


print('________________Predict and Evaluations__________________')

#use predict method off of nb to predict labels from X_test
predictions = nb.predict(X_test)

#create a confusion matrix and classification report using these predictions and y_test
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

print('________________Predict and Evaluations__________________')

#create a pipeline with the steps: CountVectorizer(),TfdifTransformer((, MultinimialNB()))

pipe = Pipeline([
		('bow',CountVectorizer()),
		('tfdif' , TfidfTransformer()),
		('model', MultinomialNB())])


print('________________Using the Pipeline__________________')

#redo the train test split 

X = yelp_class['text']
y = yelp_class['stars']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# fit the pipeline to the training data 
pipe.fit(X_train,y_train)


print('________________Predictions and Evaluation__________________')
#use the pipeline to predict from the X_test and create a classification report and confusion matrix. 

predictions = pipe.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))



