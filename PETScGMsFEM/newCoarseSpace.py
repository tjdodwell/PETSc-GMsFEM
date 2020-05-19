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

        """
            getNearestNeighbours - for each subdomain, finds the nearest neighbour processors, for which the subdomain will overlap. For each of these pairs finds corresponding degrees of freedom on each subdomain.

                self.neighbour_list - list of lists - for each subdomain contrains list of subdomains which are neighbours.

                self.dof_overlap_this - lists of list - for each subdomain which is a neighbour of 'this' processor. Will contain a lists of local dof numbers which live in the overlap

                self.dof_overlap_other - lists of list - for each subdomain which is a neighbour of 'this' processor. Will contain a lists of dof numbers on the other processor which live in the overlap.

        """

        # Initialise Lists to be constructed

        self.neighbour_list = [ [] for i in range(self.comm.Get_size()) ]

        self.dof_overlap_this = [ [] for i in range(self.comm.Get_size())]

        self.dof_overlap_other = [ [] for i in range(self.comm.Get_size())]

        # For Each Subdomain / Processor

        for i in range(self.comm.Get_size()):

            # Find Neigbours - Note all processors now global connectivity.

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

            self.neighbour_list[i] = np.nonzero(Neighbours)[0]

            # For neighbours of processor i

            for k in self.neighbour_list[i]:

                if(self.comm.Get_rank() == k):
                    # If you are the neighbour
                    workl[:] = np.arange(workl.size) + 1
                    # NB. Note use of '+1' so can use call np.nonzeros() later otherwise approach removes '0' dof from lists.
                else:
                    workl.set(0.0)

                work.set(0.0)

                self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)

                workl.set(0.0)
                self.scatter_l2g(work, workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

                if(self.comm.Get_rank() == i): # If this processor
                    id = np.asarray(np.nonzero(workl[:]), dtype = np.int32)
                    id_neighbour = np.asarray(workl[id] - 1, dtype = np.int32)
                    self.dof_overlap_this[k].append(id) # Record local nodes which will communicate with neighbour k
                    self.dof_overlap_other[k].append(id_neighbour)
