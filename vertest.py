from libs.classes import Box
import numpy as np


points = np.array([[0, 5, 8], [1, 2, 3]])
b = Box(points)

print(b.side_lengths)
