import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

if __name__ == '__main__':
    n_pts = 500
    centers = [[-1,1], [-1,-1], [1,1]]
    X, Y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)

    plt.scatter(X[Y==0,0],X[Y==0,1])
    plt.scatter(X[Y==1,0],X[Y==1,1])
    plt.scatter(X[Y==2,0],X[Y==2,1])
    plt.show()
