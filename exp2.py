"""
This examples shows our fractional lp search compared to other methods.
we choose a dataset and compare to other methods
"""

import time
import numpy as np
from keras.datasets import mnist
from sklearn.cross_validation import train_test_split
from sklearn.datasets import *
from sklearn.neighbors import NearestNeighbors
from LpSearch import LpSearch

n_queries = 3
n_hashtables=10
index_size=5
n_neighbors=1
p=0.5


def mydist(x, y):
    return np.sum(np.abs(x-y)**p)**(1/p)



# from sklearn import datasets
#dataset = load_iris()
#dataset = load_wine()
dataset = load_digits()
# #
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train=X_train.reshape(-1,784)
X_test=X_test.reshape(-1,784)



lpsearch = LpSearch(p,index_size,num_hashtables=n_hashtables)

start_time = time.time()
lpsearch.index(X_train)
end_time = time.time()
print("time elapsed for our implemntation indexing" , (end_time - start_time))



# Get exact neighbors
start_time = time.time()

nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute',
                        metric=mydist).fit(X_train)
end_time = time.time()
print("time elapsed for brute implemntation indexing" , (end_time - start_time))

count =0
for test in X_test:
    exact = nbrs.kneighbors(test.reshape(1,-1), return_distance=False)
    search = lpsearch.query(test, 3)
    if exact[0] == search[0][0]:
        count= count+1
print("precentage:",count/X_test.shape[0])

