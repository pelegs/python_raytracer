import numpy as np
from libs.classes import *


points = np.random.uniform(-2, 5, size=(3, 3))
triangle = Triangle(points)
sphere = Sphere()
ray = Ray(np.zeros(3), X_)
ray.add_hit(2.2, sphere)
ray.add_hit(1.5, triangle)
q = ray.get_closest_hit()
print(q)
