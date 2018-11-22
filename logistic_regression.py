import numpy as np
import matplotlib.pyplot as plt

## Create array of random 2D points
n_pts = 10
standard_deviation = 2
np.random.seed(0)
top_region = np.array([np.random.normal(10, standard_deviation, n_pts), np.random.normal(12, standard_deviation, n_pts)]).T
bottom_region = np.array([np.random.normal(5, standard_deviation, n_pts), np.random.normal(6, standard_deviation, n_pts)]).T

## Scatter plot
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
plt.show()
