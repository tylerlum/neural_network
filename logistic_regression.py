import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    """Draw line between x1 and x2"""
    line = plt.plot(x1, x2)
    plt.pause(0.00001)
    line[0].remove()

def sigmoid(score):
    """Calculate sigmoid of score"""
    return 1 / (1 + np.exp(-score))

def calculate_error(line_parameters, points, y):
    """Calculate cross entropy error (greater number is more error)"""
    m = points.shape[0]
    p = sigmoid(points*line_parameters)
    cross_entropy = -(np.log(p).T * y + np.log(1-p).T * (1-y)) / m
    return cross_entropy
 
def gradient_descent(line_parameters, points, y, alpha, num_iterations = 5000):
    m = points.shape[0]
    for i in range(0, num_iterations):
        ## Calculate gradient
        p = sigmoid(points*line_parameters)
        gradient = (points.T * (p-y)) * alpha / m

        ## Update line parameters
        line_parameters = line_parameters - gradient 
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        ## Recalculate line
        x1 = np.array([points[:,0].min(), points[:,0].max()])
        x2 = -(b/w2) + x1 * (-w1/w2)

        ## Draw each line 
        draw(x1, x2)
        
## Create array of random 2D points
n_pts = 30 
standard_deviation = 2
np.random.seed(1)

bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, standard_deviation, n_pts), np.random.normal(12, standard_deviation, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, standard_deviation, n_pts), np.random.normal(6, standard_deviation, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

## Define line parameters
line_parameters = np.matrix([np.zeros(3)]).T

## Label outputs
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(2*n_pts,1)

## Scatter plot
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
gradient_descent(line_parameters, all_points, y, 0.06)
plt.show() 

## Find probabilities of being blue
linear_combination = all_points * line_parameters
probabilities = sigmoid(linear_combination)

## Calculate Error
cross_entropy = calculate_error(line_parameters, all_points, y)

