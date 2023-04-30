import bpy
import numpy as np


coords = np.load("/home/pelegs/prog/python/triangle_tests/coords.npy")
for p in coords:
    bpy.ops.mesh.primitive_uv_sphere_add(location=p, radius=0.5)
