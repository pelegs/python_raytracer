from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from subprocess import run


run("rm -rf libs", shell=True)
run("mkdir libs", shell=True)
run("rm -rf ../build", shell=True)
run("rm cymath_new.c", shell=True)
run("rm cymath_new.cpython*.so", shell=True)

setup(
    name="Math functions",
    ext_modules=cythonize(
        "cymath_new.pyx",
        include_path=[np.get_include()],
        annotate=True,
    ),
    include_dirs=[np.get_include()],  # Include directory not hard-wired
)

run("mv libs/cymath_new.cpython*.so .", shell=True)
run("mv build ../", shell=True)
