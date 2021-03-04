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
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

df = pd.read_csv('College_Data')
print(df)
print(df.info())
print(df.describe())

print('_____________________Exploratory Data Analysis_____________________________')

#create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column
sns.lmplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private',fit_reg=False,palette='coolwarm',height=6,aspect=1)


#create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column
sns.lmplot(x='Outstate',y='F.Undergrad',data=df,hue='Private',fit_reg=False,height=6,aspect=1)


#create a stacked histogram showing out of statte tuition based on the Private colummn. Try doing this using sns.FacetGrid.
g = sns.FacetGrid(df,hue='Private',palette='coolwarm')
g = g.map(plt.hist,'Outstate', bins=20,alpha=0.7)


#Create a similar hstogram for the Grad.Rate column
g = sns.FacetGrid(df,hue='Private',palette='coolwarm')
g = g.map(plt.hist,'Grad.Rate', bins=20,alpha=0.7)
#plt.show()

#Notice how there seems to be a private school with a gradutaion rate of higher than 100% what is the name of that school?
print(df[df['Grad.Rate']>100])

df['Grad.Rate']['Cazenovia College'] = 100


print('_____________________K Means Cluster Creation____________________________')

kmeans = KMeans(n_clusters=2)
#kmeans.fit(df.drop('Private',axis=1))

#print(kmeans.cluster_centers_)
#print(kmeans.labels_)

#create a new column dor fd called 'Cluster' which is a 1 for a Private school, and a 0 for a public school

def converter(private):
	if private == 'Yes':
		return 1
	else: 
		return 0 

df['Cluster'] = df['Private'].apply(converter)
print(df)

print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(f['Cluster'],kmeans.labels_))


