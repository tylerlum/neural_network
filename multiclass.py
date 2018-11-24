import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

## Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

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
    pred_func = model.predict_classes(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
 
if __name__ == '__main__':
    ## Setup dataset
    n_pts = 500
    centers = [[-1,1], [-1,-1], [1,1], [1,-1], [0,0]]
    X, Y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)

    ## Plot datasets
    plt.scatter(X[Y==0,0],X[Y==0,1])
    plt.scatter(X[Y==1,0],X[Y==1,1])
    plt.scatter(X[Y==2,0],X[Y==2,1])
    plt.scatter(X[Y==3,0],X[Y==3,1])
    plt.scatter(X[Y==4,0],X[Y==4,1])
    # plt.show()

    ## One hot encoding
    Y_cat = to_categorical(Y, 5) 

    ## Make neural network
    model = Sequential()
    model.add(Dense(units=5, input_shape=(2,), activation='softmax'))
    model.compile(Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=X, y=Y_cat, verbose=1, batch_size=50, epochs=100)

    ## Plot 
    plot_decision_boundary(X, Y_cat, model)
    plt.scatter(X[Y==0,0],X[Y==0,1])
    plt.scatter(X[Y==1,0],X[Y==1,1])
    plt.scatter(X[Y==2,0],X[Y==2,1])
    plt.scatter(X[Y==3,0],X[Y==3,1])
    plt.scatter(X[Y==4,0],X[Y==4,1])
     
    ## Predict point
    x, y = 0.5, 0.5
    point = np.array([[x,y]])
    prediction = model.predict(point)
    plt.plot([x],[y], marker='o', markersize=10, color='r')
    plt.show()
    print("Point {0}: Prediction is {1}".format(point[0], prediction))
