import numpy as np
import matplotlib.pyplot as plt

## Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def plot_decision_boundary(X, Y, model):
    """Make a plot of the decision boundary"""
    ## Span of x and y values of points
    tolerance = 1
    xa_span = np.linspace(min(X[:,0]) - tolerance, max(X[:,0]) + tolerance)
    xb_span = np.linspace(min(X[:,1]) - tolerance, max(X[:,1]) + tolerance)
    
    ## Make meshgrid
    xx, yy = np.meshgrid(xa_span, xb_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
 
def plot_history(h, y_variable):
    """Plot of y_variable vs. epoch. eg. 'acc', 'loss'"""
    plt.plot(h.history[y_variable])
    plt.title(y_variable)
    plt.xlabel('epoch')
    plt.legend([y_variable])
    plt.show()

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
# plt.show()

## Create perceptron
model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
adam = Adam(lr = 0.1)
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs=300, shuffle='true')

## Plot accuracy
# plot_history(h, 'acc')

## Plot loss
# plot_history(h, 'loss')

## Plot decision boundary and training data points
plot_decision_boundary(X, Y, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

## Get predict new data
x, y = 7.5, 5
point = np.array([[x, y]])
print(point)
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color='b')
print("Prediction is {0}".format(prediction))
plt.show()
