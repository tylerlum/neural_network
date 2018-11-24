import numpy as np
import matplotlib.pyplot as plt

## Datasets
from sklearn import datasets

## Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from perceptron import plot_history, plot_decision_boundary


if __name__ == '__main__':
    np.random.seed(0)
    
    ## Make circle of points
    n_pts = 500
    X, Y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
    
    ## Plot circle of points
    # plt.scatter(X[Y==0, 0], X[Y==0, 1])
    # plt.scatter(X[Y==1, 0], X[Y==1, 1])
    # plt.show()
    
    ## Make model
    model = Sequential()
    model.add(Dense(4, input_shape=(2,), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])
    h = model.fit(x=X, y=Y, verbose=1, batch_size=20, epochs=100, shuffle='true')
    
    ## Plot accuracy
    plot_history(h, 'acc')
    plt.show()
    
    plot_history(h, 'loss')
    plt.show()
    
    ## Plot decision boundary
    plot_decision_boundary(X, Y, model)
    plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
    plt.scatter(X[n_pts:, 0], X[n_pts:, 1], color = 'r')

    ## Test prediction of point
    x0, y0, x1, y1 = 0, 0, 1.5, 2
    point0 = np.array([[x0,y0]])
    point1 = np.array([[x1,y1]])

    prediction0 = model.predict(point0)
    prediction1 = model.predict(point1)

    plt.plot([x0], [y0], marker='o', markersize=10, color='r')
    plt.plot([x1], [y1], marker='o', markersize=10, color='g')
    plt.show()

    print("Point ({0}, {1}): Prediction is {2}".format(x0, y0, prediction0))
    print("Point ({0}, {1}): Prediction is {2}".format(x1, y1, prediction1))
