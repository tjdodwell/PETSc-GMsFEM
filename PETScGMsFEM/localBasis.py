from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

import time

from PETScGMsFEM import *


class localBasis():


    def __init__(self):

        print("Hello World")
