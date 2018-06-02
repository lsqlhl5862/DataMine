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

df_train = pd.read_csv("test.csv")
#for(i in range(df_train["DIRECTION"])):
print(len(df_train["TERMINALNO"].unique()))