from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

import time

class PartitionOfUnity():

    # PartitionOfUnity - class defines partition of untity operator.

    def __init__(self, da, A, A_local, comm, scatter_l2g):

        self.da = da

        self.A = A

        self.A_local = A_local

        self.comm = comm

        self.scatter_l2g = scatter_l2g

    def build(self, plotPOU = True):

        # Build Partition of Unity Operator - Compute the multiplicity of each degree of freedom and max of the multiplicity

        xi_local,_ = self.A_local.getVecs()
        xi_global,_ = self.A.getVecs()

        xi_local.set(1.)
        xi_global.set(0.)
        self.scatter_l2g(xi_local, xi_global, PETSc.InsertMode.ADD_VALUES)
        self.scatter_l2g(xi_global, xi_local, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)


        for i in range(xi_local[:].size):
            xi_local[i] = 1.0 / xi_local[i]


        self.X = self.A_local.copy()
        self.X.zeroEntries()
        self.X.setDiagonal(xi_local)


        if(plotPOU):
            viewer = PETSc.Viewer().createVTK('POU_{}.vts'.format(self.comm.Get_rank()), 'w', comm = PETSc.COMM_WORLD)
            xi_global.view(viewer)
            tmp = self.da.createGlobalVec()
            tmpl_a = self.da.getVecArray(tmp)
            work_a = self.da.getVecArray(xi_global)
            tmpl_a[:] = work_a[:]
            tmp.view(viewer)
            viewer.destroy()
