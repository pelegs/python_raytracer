from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from subprocess import run


run("rm -rf libs", shell=True)
run("mkdir libs", shell=True)
run("rm -rf ../build", shell=True)
run("rm cymath.c", shell=True)
run("rm cymath.cpython*.so", shell=True)

setup(
    name="Math functions",
    ext_modules=cythonize(
        "*.pyx",
        include_path=[np.get_include()],
    ),
    include_dirs=[np.get_include()],  # Include directory not hard-wired
)

run("mv libs/cymath.cpython*.so .", shell=True)
run("mv build ../", shell=True)
