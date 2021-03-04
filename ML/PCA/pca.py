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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'],columns = cancer['feature_names'])

print(cancer['target_names'])

scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

#PCA 

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
#plt.show()

print(pca.components_)
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
print(df_comp)

sns.heatmap(df_comp,cmap='plasma')
plt.show()
