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


        self.size = 0

        self.basis = []

        self.globalID = []

        self.isBnd = []



    def setGlobalID(self, id_local, id_global):
        """
            setGlobalID - sets the global id of local basis j
        """
        self.globalID[id_local] = id_global


    def add(self, vec, id_global = None, isBoundary = False):

        """
            add() - adds a single vec to the basis

        """

        self.basis.append(vec)

        self.globalID.append(id_global)

        self.isBnd.append(isBoundary)

        self.size += 1 # Increment local basis dimension
