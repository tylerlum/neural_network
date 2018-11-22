import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    """Draw line between x1 and x2"""
    line = plt.plot(x1, x2)

## Create array of random 2D points
n_pts = 100
standard_deviation = 2
np.random.seed(0)

bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, standard_deviation, n_pts), np.random.normal(12, standard_deviation, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, standard_deviation, n_pts), np.random.normal(6, standard_deviation, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

## Define line parameters
w1 = -0.2
w2 = -0.35
b = 3.5
line_parameters = np.matrix([w1, w2, b])

## Find upper right and lower left points to start line
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -(b/w2) + x1 * (-w1/w2)

## Scatter plot
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
draw(x1, x2)
plt.show()
