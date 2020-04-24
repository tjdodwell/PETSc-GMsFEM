from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name = "PETScGMsFEM",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
        'matplotlib',
        'setuptools',
        'cython',
        'mpi4py',
        'petsc4py',
        'slepc4py',
        'vtk',
        'pytest'
        ]


        ]
    #ext_modules = cythonize('GenEO/matelem_cython.pyx'),
)
