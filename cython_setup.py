from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="Math functions",
    ext_modules= cythonize(
            "*.pyx", include_path=[np.get_include()],
        ),
    include_dirs=[np.get_include()],  # Include directory not hard-wired
)
