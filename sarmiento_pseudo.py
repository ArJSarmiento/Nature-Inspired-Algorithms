# %matplotlib inline
import random

import matplotlib.pyplot as plt
import numpy as np

lb_x = -10.0
ub_x = 100000
lb_y = -1000000
ub_y = 10.0


def f(x, y):
    term1 = np.sin(3 * np.pi * x) ** 2
    term2 = (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
    term3 = (y - 1) ** 2 * (1 + np.sin(2 * np.pi * y) ** 2)
    return term1 + term2 + term3


def generate_value(lb, ub):
    return lb + (ub - lb) * random.random()


# create a contour3d map with equally space x and y values #for comparison
x = np.linspace(lb_x, ub_x, 150)
y = np.linspace(lb_y, ub_y, 150)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.contour3D(X, Y, Z, 80, cmap='hot')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f')

xline = []
yline = []
zline = []

population_size = 1000

# generate pseudo-random values for × and y
for i in range(population_size):
    xline.append(generate_value(lb_x, ub_x))
    yline.append(generate_value(lb_y, ub_y))

# get the equivalent function value for each pair (x,y)
for i in range(population_size):
    zline.append(f(xline[i], yline[i]))

# convert lists to numpy array for scatter3D
xline = np.array(xline)
yline = np.array(yline)
zline = np.array(zline)

# graph the function
axl = plt.axes(projection='3d')
axl.scatter3D(xline, yline, zline, c=zline, cmap='hot', s=7)
axl.set_xlabel('x')
axl.set_ylabel('y')
axl.set_zlabel('f')
