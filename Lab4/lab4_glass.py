import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from sklearn import datasets
plt.style.use('ggplot')

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA  
from time import time   

df_train = pd.read_csv("glass.test", skiprows=1, header=None)

df = pd.read_csv("glass.data", skiprows=1, header=None)
df[9]=LabelEncoder().fit_transform(df[9].values)

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
x_train=df_train.iloc[:,:9]
x_test=df.iloc[:,:9]
y_test=df[9].tolist()

# K-Means
clf = KMeans(n_clusters=6, random_state=random_state)

clf.fit(x_train)
y_pred=clf.predict(x_test)


x=PCA(n_components = 2).fit_transform(x_test)

plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("K-Means")

plots = []
names = []
result = {}
sizes = np.arange(2,20,2)
for size in sizes:
    clf = KMeans(n_clusters=size, random_state=random_state)
    clf.fit(x_train)
    y_pred=clf.predict(x_test)
    adjusted_rand_score     = metrics.adjusted_rand_score(y_test,y_pred)
    v_measure_score     = metrics.v_measure_score(y_test,y_pred)
    adjusted_mutual_info_score     = metrics.adjusted_mutual_info_score(y_test,y_pred)
    mutual_info_score     = metrics.mutual_info_score(y_test,y_pred)
    calinski_harabaz_score = metrics.calinski_harabaz_score(x_test, y_pred)/100
    result[size] = (adjusted_rand_score,v_measure_score,adjusted_mutual_info_score,mutual_info_score,calinski_harabaz_score)

result = pd.DataFrame(result).transpose()
result.columns = ['adjusted_rand_score','v_measure_score','adjusted_mutual_info_score','mutual_info_score','calinski_harabaz_score']
result.plot(marker='*', figsize=(15,5))
plt.xlabel('Size of n_clusters')
plt.ylabel('Score')
plt.title("K-Means")
plt.show()


# Affinity propagation
clf=AffinityPropagation(preference=-30)
clf.fit(x_train)
y_pred=clf.predict(x_test)

x=PCA(n_components = 2).fit_transform(x_test)


plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("Affinity propagation")

plots = []
names = []
result = {}
sizes = np.arange(-5,-50,-5)
for size in sizes:
    clf = AffinityPropagation(preference=-30)
    clf.fit(x_train)
    y_pred=clf.predict(x_test)
    adjusted_rand_score     = metrics.adjusted_rand_score(y_test,y_pred)
    v_measure_score     = metrics.v_measure_score(y_test,y_pred)
    adjusted_mutual_info_score     = metrics.adjusted_mutual_info_score(y_test,y_pred)
    mutual_info_score     = metrics.mutual_info_score(y_test,y_pred)
    calinski_harabaz_score = metrics.calinski_harabaz_score(x_test, y_pred)/100
    result[size] = (adjusted_rand_score,v_measure_score,adjusted_mutual_info_score,mutual_info_score,calinski_harabaz_score)

result = pd.DataFrame(result).transpose()
result.columns = ['adjusted_rand_score','v_measure_score','adjusted_mutual_info_score','mutual_info_score','calinski_harabaz_score']
result.plot(marker='*', figsize=(15,5))
plt.xlabel('Size of preference')
plt.ylabel('Score')
plt.title("Affinity propagation")
plt.show()

# DBSCAN
clf=DBSCAN(eps=0.8, min_samples=16)
clf.fit(x_train)
y_pred=clf.fit_predict(x_test)

x=PCA(n_components = 2).fit_transform(x_test)

plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("DBSCAN")

plots = []
names = []
result = {}
sizes = np.arange(1,10,1)
for size in sizes:
    clf = DBSCAN(eps=size/10, min_samples=4)
    clf.fit(x_train)
    y_pred=clf.fit_predict(x_test)
    adjusted_rand_score     = metrics.adjusted_rand_score(y_test,y_pred)
    v_measure_score     = metrics.v_measure_score(y_test,y_pred)
    adjusted_mutual_info_score     = metrics.adjusted_mutual_info_score(y_test,y_pred)
    mutual_info_score     = metrics.mutual_info_score(y_test,y_pred)
    result[size] = (adjusted_rand_score,v_measure_score,adjusted_mutual_info_score,mutual_info_score)

result = pd.DataFrame(result).transpose()
result.columns = ['adjusted_rand_score','v_measure_score','adjusted_mutual_info_score','mutual_info_score']
result.plot(marker='*', figsize=(15,5))
plt.xlabel('Size of eps')
plt.ylabel('Score')
plt.title("DBSCAN")
plt.show()
#AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
clf=AgglomerativeClustering(n_clusters=6,linkage='ward')
clf.fit(x_train)
y_pred=clf.fit_predict(x_test)

x=PCA(n_components = 2).fit_transform(x_test)


plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("AgglomerativeClustering")
plots = []
names = []
result = {}
sizes = np.arange(2,20,2)
for size in sizes:
    clf = AgglomerativeClustering(n_clusters=size,linkage='ward')
    clf.fit(x_train)
    y_pred=clf.fit_predict(x_test)
    adjusted_rand_score     = metrics.adjusted_rand_score(y_test,y_pred)
    v_measure_score     = metrics.v_measure_score(y_test,y_pred)
    adjusted_mutual_info_score     = metrics.adjusted_mutual_info_score(y_test,y_pred)
    mutual_info_score     = metrics.mutual_info_score(y_test,y_pred)
    calinski_harabaz_score = metrics.calinski_harabaz_score(x_test, y_pred)/100
    result[size] = (adjusted_rand_score,v_measure_score,adjusted_mutual_info_score,mutual_info_score,calinski_harabaz_score)

result = pd.DataFrame(result).transpose()
result.columns = ['adjusted_rand_score','v_measure_score','adjusted_mutual_info_score','mutual_info_score','calinski_harabaz_score']
result.plot(marker='*', figsize=(15,5))
plt.xlabel('Size of n_clusters')
plt.ylabel('Score')
plt.title("AgglomerativeClustering")
plt.show()
