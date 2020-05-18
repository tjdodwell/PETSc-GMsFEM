from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

from pyevtk.hl import unstructuredGridToVTK

from pyevtk.hl import pointsToVTK

import time

from PETScGMsFEM import *

comm = mpi.COMM_WORLD


class SimplestModel():

    def __init__(self, n, L, proc, overlap, comm):

        self.comm = comm
        self.n = n
        self.L = L
        self.proc = proc
        self.overlap = overlap
        self.dim = len(L)

        # Build Grid
        self.da = PETSc.DMDA().create(n, dof=1, proc_sizes = proc, stencil_width=overlap)
        self.da.setUniformCoordinates(xmax=L[0], ymax=L[1], zmax=L[2])
        self.da.setMatType(PETSc.Mat.Type.AIJ)

        elem = self.da.getElements()
        self.nel = elem.shape[0]

        # Setup boundary condition

        self.isBnd = lambda x: self.isBoundary(x)

        # Define Finite Element space

        # self.fe = DarcyQ1(self.dim)

        # Setup global and local matrices + communicators

        self.A = self.da.createMatrix()
        r, _ = self.A.getLGMap() # Get local to global mapping
        self.is_A = PETSc.IS().createGeneral(r.indices) # Create Index Set for local indices
        A_local = self.A.createSubMatrices(self.is_A)[0] # Construct local submatrix on domain
        vglobal = self.da.createGlobalVec()
        vlocal = self.da.createLocalVec()
        self.scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, self.is_A)

        self.A_local = A_local

        # Identify boundary nodes

        self.nnodes = int(self.da.getCoordinatesLocal()[ :].size/self.dim)
        self.coordsLocal = np.transpose(self.da.getCoordinatesLocal()[:].reshape((self.nnodes,self.dim)))
        Dirich, Neumann, P2P = checkFaces(self.da, self.isBnd, self.coordsLocal)

        # Initialise Coarse Space

        self.VH = newCoarseSpace(self.da,self.comm)

    def isBoundary(self, x):
        val = 0.0
        if(x[0] < 1e-6):
            output = True
            val = 0.0
            type = 1
        elif(x[0] > self.L[0] - 1e-6):
            output = True
            val = 1.0
            type = 1
        elif((x[1] < 1e-6) or (x[2] < 1e-6) or (x[1] > self.L[1] - 1e-6) or (x[2] > self.L[2] - 1e-6)):
            type = 2 # Neumann Boundary
        else:
            type = 0 # Either internal or processor to processor boundary

        return val, type




def main():

    n = [9, 2, 2]

    proc = [3, 1, 1]

    L = [3.0, 1.0, 1.0]

    overlap = 1

    mkl = 512

    myModel = SimplestModel(n, L, proc, overlap, comm)

    # First set of test check that neighbour communication is correct

    # Test 1 - Find Neighbours

    if(comm.Get_rank() == 0):
        print("*** Testing Neigbour lists ...", end = "  ")
    assert myModel.VH.neighbour_list[0][0] == 1, "Neighbour list is not as expected on processor 0"
    assert myModel.VH.neighbour_list[1][0] == 0, "Neighbour list is not as expected on processor 1"
    assert myModel.VH.neighbour_list[1][1] == 2, "Neighbour list is not as expected on processor 1"
    assert myModel.VH.neighbour_list[2][0] == 1, "Neighbour list is not as expected on processor 2"
    if(comm.Get_rank() == 0):
        print("PASS")

    # Test 2 -

    if(comm.Get_rank() == 0):

        print("*** Testing Neighbour to Neighbour Dofs", end = "  ")

        # Find coordinates of overlap

        overlap_x = myModel.coordsLocal[0,myModel.VH.dof_proc[1]]

        data_send = myModel.VH.dof_other[1][0][0]

        comm.Isend(data_send, dest = 1)#, tag = self.generateTag(vec[this], vec[other], j))

        comm.Isend(overlap_x, dest = 1)

    elif(comm.Get_rank() == 1):

        localNumber = myModel.VH.dof_proc[0][0][0]

        data_recv = np.zeros(len(localNumber), dtype=np.int32)

        overlap_x = np.zeros(len(localNumber), dtype=np.float64)

        req = comm.Irecv(data_recv, source = 0)

        req = comm.Irecv(overlap_x, source = 0)

    comm.barrier()

    if(comm.Get_rank() == 1):
        data = np.zeros(1, dtype='i')
    else:
        data = np.empty(1, dtype='i')

    if(comm.Get_rank() == 1):

        print(np.sum(np.abs(myModel.coordsLocal[0,data_recv] - overlap_x)))

        if(np.sum(np.abs(myModel.coordsLocal[0,data_recv] - overlap_x)) < 1e-4):

            data[0] = 1

    comm.Bcast(data, root = 1)
    comm.barrier()

    if(comm.Get_rank() == 0):
        if(data[0]== 1):
            print("PASS")
        else:
            print("FAIL")


    # Receive from other processor
"""
    start = np.sum(self.localSizes[0:vec[other]])

    for j in range(self.localSizes[vec[other]]): # For all the basis functions on the other subdaomin

        id_local = myModel.VH.dof_proc[]

        data_tmp = np.empty(len(id), dtype=np.float64)

        req = self.comm.Irecv(data_tmp, source=vec[other])#, tag = self.generateTag(vec[other], vec[this], j))



    elif(comm.Get_rank() == 0):

        overlap_x = myModel.coordsLocal[:,myModel.VH.dof_other[0]]

        print("This is processor = 0" + str(overlap_x[0][0][0]))






"""







if __name__ == "__main__":
    main()
