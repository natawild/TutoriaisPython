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

data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)

plt.scatter(data[0][:,0],data[0][:,1])

plt.scatter(data[0][:,0],data[0][:,1],c=data[1])
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

print(kmeans.cluster_centers_)
print(kmeans.labels_)

fig, (ax1,ax2) = plt.subplot = plt.subplots(1,2,sharey=True,figsize=(10,6))
ax1.set_title('K means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

plt.show()


