
import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
plt.style.use('ggplot')


df = pd.read_table("mushroom.dat", header=None, delim_whitespace=True)

from sklearn.tree import DecisionTreeClassifier
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

def naive_split(X, Y, n):
    # Take first n lines of X and Y for training and the rest for testing
    X_train = X[:n]
    X_test  = X[n:]
    Y_train = Y[:n]
    Y_test  = Y[n:]
    return (X_train, X_test, Y_train, Y_test)

def train_model(n=7000):
    # Given X_dummy and Y_dummy, split naively into training and testing sets
    X_train, X_test, Y_train, Y_test = naive_split(X, y, n)
    # Instantiate a default decision tree with fixed random state
    # NOTE: In real life you'd probably want to remove the fixed seed.
    clf = tree.DecisionTreeClassifier(random_state=42)
    # Next, train a default decision tree using the training sets
    clf.fit(X_train, Y_train)
    # Lastly, return the test sets and the trained model
    return (X_test, Y_test, clf)

import sklearn.metrics as metrics

sizes = np.arange(100,len(y), 500)
result = {}
for size in sizes:
    X_test, Y_test, clf = train_model(n=size)
    score     = clf.score(X_test, Y_test)
    precision = metrics.precision_score(Y_test, clf.predict(X_test))
    recall    = metrics.recall_score(Y_test, clf.predict(X_test))
    result[size] = (score, precision, recall)
# Turn the results into a DataFrame
result = pd.DataFrame(result).transpose()
result.columns = ['Accuracy', 'Precision', 'Recall']
result.plot(marker='*', figsize=(15,5))
plt.xlabel('Size of training set')
plt.ylabel('Value')
plt.show()



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X, y)

def train_model_knn(n=7000):
    # Given X_dummy and Y_dummy, split naively into training and testing sets
    X_train, X_test, Y_train, Y_test = naive_split(X, y, n)
    # Instantiate a default decision tree with fixed random state
    # NOTE: In real life you'd probably want to remove the fixed seed.
    knn = KNeighborsClassifier()
    # Next, train a default decision tree using the training sets
    knn.fit(X_train, Y_train)
    # Lastly, return the test sets and the trained model
    return (X_test, Y_test, knn)


sizes = np.arange(100,len(y), 500)
result = {}
for size in sizes:
    X_test, Y_test, knn = train_model_knn(n=size)
    score     = knn.score(X_test, Y_test)
    precision = metrics.precision_score(Y_test, knn.predict(X_test))
    recall    = metrics.recall_score(Y_test, knn.predict(X_test))
    result[size] = (score, precision, recall)
# Turn the results into a DataFrame
result = pd.DataFrame(result).transpose()
result.columns = ['Accuracy', 'Precision', 'Recall']
result.plot(marker='*', figsize=(15,5))
plt.xlabel('Size of training set')
plt.ylabel('Value')
plt.show()