import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from libs.cymath import *
from libs.classes_cy import *


def rays_to_sphere(c, r, dth):
    px, py = c[:2]
    l2 = np.sqrt(px**2+py**2-r**2)
    th = np.arctan2(py, px)
    th2 = np.arctan2(r, l2)
    d = np.zeros((2, 3))
    d[0] = l2*np.array([np.cos(th+th2-dth), np.sin(th+th2-dth), 0.0])
    d[1] = l2*np.array([np.cos(th-th2+dth), np.sin(th-th2+dth), 0.0])
    return d.T


c = np.random.uniform(-7, 7, size=2)
c = np.insert(c, 2, 0.0)
r = np.random.uniform(1, 0.75*np.sqrt(c[0]**2+c[1]**2))

d = rays_to_sphere(c, r, dth=0.1)
d0_unit = unit(d.T[0])
d1_unit = unit(d.T[1])

t0 = line_sphere_intersection(O_, d0_unit, c, r)
l0 = Line(O_, d0_unit)
p0 = l0.at_point(t0)
t1 = line_sphere_intersection(O_, d1_unit, c, r)
l1 = Line(O_, d1_unit)
p1 = l1.at_point(t1)
print(t0, p0)


fig, ax = plt.subplots(figsize=(15, 15))
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
circ = plt.Circle(c[:2], r, color="red")
ax.add_patch(circ)
w, h = 0.3, 0.3
rect_wh_half = np.array([w, h])/2
rect0 = patches.Rectangle(p0[:2]-rect_wh_half, w, h, linewidth=2, facecolor='purple')
ax.add_patch(rect0)
rect1 = patches.Rectangle(p1[:2]-rect_wh_half, w, h, linewidth=2, facecolor='purple')
ax.add_patch(rect1)
plt.plot((0, d[0,0]), (0, d[1,0]), 'bo-')
plt.plot((0, d[0,1]), (0, d[1,1]), 'bo-')
plt.plot((0, d0_unit[0]), (0, d0_unit[1]), 'go-')
plt.plot((0, d1_unit[0]), (0, d1_unit[1]), 'go-')
fig.savefig("pics/geotest1.png")
