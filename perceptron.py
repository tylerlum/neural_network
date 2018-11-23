import numpy as np
import matplotlib.pyplot as plt

## Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

## Make random 2D arrays
n_pts = 500
np.random.seed(0)
standard_deviation = 2
Xa = np.array([np.random.normal(13, standard_deviation, n_pts), np.random.normal(12, standard_deviation, n_pts)]).T
Xb = np.array([np.random.normal(8, standard_deviation, n_pts), np.random.normal(6, standard_deviation, n_pts)]).T

X = np.vstack((Xa, Xb))
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

## Scatter plot
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
plt.show()

## Create perceptron
model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
adam = Adam(lr = 0.1)
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs=500, shuffle='true')
