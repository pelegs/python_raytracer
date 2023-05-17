import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from libs.cymath import *

# Fixing random state for reproducibility

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 1000
dir = unit(np.random.uniform(-1, 1, size=3))
points = np.zeros((n, 3), dtype=np.double)
for i, vec in enumerate(points):
    points[i] = rand_pt_solid_angle_rotated(
        dir, th=np.pi/5
    )

xs = points[:,0]
ys = points[:,1]
zs = points[:,2]

ax.scatter(xs, ys, zs)
ax.axes.set_xlim3d([-2, 2])
ax.axes.set_ylim3d([-2, 2])
ax.axes.set_zlim3d([-2, 2])
ax.set_aspect('equal')
u, v, w = dir
ax.quiver(0., 0., 0., u, v, w)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
