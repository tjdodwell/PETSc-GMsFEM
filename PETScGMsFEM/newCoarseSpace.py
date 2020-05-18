from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

import time

from PETScGMsFEM import *

class newCoarseSpace():

    def __init__(self, da, comm):

        self.da = da # Store Structured Grid

        self.comm = comm

        # Set up Matrices

        self.A = self.da.createMatrix()

        r, _ = self.A.getLGMap() # Get local to global mapping

        self.is_A = PETSc.IS().createGeneral(r.indices) # Create Index Set for local indices

        self.A_local = self.A.createSubMatrices(self.is_A)[0] # Construct local submatrix on domain

        vglobal = self.da.createGlobalVec()
        vlocal = self.da.createLocalVec()

        self.scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, self.is_A)

        # Define a local basis

        self.Phi = localBasis()

        # Define local Partition of Unity Operator

        self.POU = PartitionOfUnity(self.da, self.A, self.A_local, self.comm, self.scatter_l2g)

        self.POU.build(True)

        # Define Nearest Neighbour Communication

        self.getNearestNeighbours()

    def getNearestNeighbours(self):

        self.neighbour_list = []

        self.overlap_dofs_send = [ ]
        self.overlap_dofs_recv = [ ]

        # dof_proc - list of dofs belong to proc which will shared with other processors

        self.dof_proc = [ [] for i in range(self.comm.Get_size())]

        # dof_other - list of dofs belong to neighbour proc which will shared with other processors

        self.dof_other = [ [] for i in range(self.comm.Get_size())]

        for i in range(self.comm.Get_size()): # For each processor

            work = self.da.createGlobalVec()
            workl = self.da.createLocalVec()

            if(self.comm.Get_rank() == i):
                workl.set(1.0)
            else:
                workl.set(0.0)
            work.set(0.0)

            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)

            workl.set(0.0)
            self.scatter_l2g(work, workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

            isNeighbour = np.zeros(1, dtype = np.int32)

            Neighbours = np.zeros(self.comm.Get_size(), dtype = np.int32)

            if(np.sum(workl[:]) > 0 and (self.comm.Get_rank() != i)):
                isNeighbour[0] = 1 # This is a nearest neigbour to processor i
                id = np.asarray(np.nonzero(workl[:]), dtype = np.int32)

            self.comm.Allgather([isNeighbour, mpi.INT], [Neighbours, mpi.INT])

            self.neighbour_list.append(np.nonzero(Neighbours)[0])

            # All Processors know about global connectivity

            for k in self.neighbour_list[-1]: # For the neighbours of processor i

                if(self.comm.Get_rank() == k): # If you are the neighbour
                    workl[:] = np.arange(workl.size) + 1
                else:
                    workl.set(0.0)

                work.set(0.0)

                self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)

                workl.set(0.0)
                self.scatter_l2g(work, workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

                if(self.comm.Get_rank() == i): # If this processor
                    id = np.asarray(np.nonzero(workl[:]), dtype = np.int32)

                    #Issue with is non-zero removes '0' dof from the list.
                    id_neighbour = np.asarray(workl[id] - 1, dtype = np.int32)
                    self.dof_proc[k].append(id) # Record local nodes which will communicate with neighbour k
                
                    self.dof_other[k].append(id_neighbour)
