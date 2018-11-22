import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    """Draw line between x1 and x2"""
    line = plt.plot(x1, x2)

def sigmoid(score):
    """Calculate sigmoid of score"""
    return 1 / (1 + np.exp(-score))

def calculate_error(line_parameters, points, y):
    """Calculate cross entropy error (greater number is more error)"""
    m = points.shape[0]
    p = sigmoid(points*line_parameters)
    cross_entropy = -(np.log(p).T * y + np.log(1-p).T * (1-y)) / m
    return cross_entropy
 
## Create array of random 2D points
n_pts = 7 
standard_deviation = 2
np.random.seed(0)

bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, standard_deviation, n_pts), np.random.normal(12, standard_deviation, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, standard_deviation, n_pts), np.random.normal(6, standard_deviation, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

## Define line parameters
w1 = -0.1
w2 = -0.15
b = 0 
line_parameters = np.matrix([w1, w2, b]).T

## Find upper right and lower left points to start line
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -(b/w2) + x1 * (-w1/w2)

## Scatter plot
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
draw(x1, x2)
plt.show() 

## Find probabilities of being blue
linear_combination = all_points * line_parameters
probabilities = sigmoid(linear_combination)

## Label outputs
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(2*n_pts,1)

## Calculate Error
cross_entropy = calculate_error(line_parameters, all_points, y)
print(cross_entropy)
