import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from libs.cymath import *

# Fixing random state for reproducibility

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 1000
""" points = rand_pt_sphere_surface(n) """
points = np.zeros((n, 3), dtype=np.double)
for i, vec in enumerate(points):
    points[i] = rand_pt_solid_angle(th=np.pi/2)

xs = points[:,0]
ys = points[:,1]
zs = points[:,2]

ax.scatter(xs, ys, zs)
ax.axes.set_xlim3d([-2, 2])
ax.axes.set_ylim3d([-2, 2])
ax.axes.set_zlim3d([-2, 2])
""" u, v, w = base_vec """
""" ax.quiver(0., 0., 0., u, v, w) """

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
