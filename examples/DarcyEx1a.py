from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from PETScGMsFEM import *

comm = MPI.COMM_WORLD

if(comm.Get_rank() == 0):
    print("Running Grid Test.py")

class DarcyEx1:

    def __init__(self, n, L, overlap, comm):

        # Build Grid

        self.dim = len(n)

        self.L = L

        self.comm = comm

        self.isBnd = lambda x: self.isBoundary(x)

        self.da = PETSc.DMDA().create(n, dof=1, stencil_width=overlap)
        self.da.setUniformCoordinates(xmax=L[0], ymax=L[1], zmax=L[2])
        self.da.setMatType(PETSc.Mat.Type.AIJ)

        self.numSub = comm.Get_size()

        self.sub2proc = [ [] for i in range(self.numSub) ]
        for i in range(self.numSub):
            self.sub2proc[i] = [i, 0]

        self.M = self.numSub

        # Define Finite Element space

        self.fe = DarcyQ1(self.dim)

        # Setup global and local matrices + communicators

        self.A = self.da.createMatrix()
        r, _ = self.A.getLGMap() # Get local to global mapping
        self.is_A = PETSc.IS().createGeneral(r.indices) # Create Index Set for local indices
        A_local = self.A.createSubMatrices(self.is_A)[0] # Construct local submatrix on domain
        vglobal = self.da.createGlobalVec()
        vlocal = self.da.createLocalVec()
        self.scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, self.is_A)

        # Identify boundary nodes

        nnodes = int(self.da.getCoordinatesLocal()[ :].size/self.dim)
        coords = np.transpose(self.da.getCoordinatesLocal()[:].reshape((nnodes,self.dim)))
        Dirich, Neumann, P2P = checkFaces(self.da, self.isBnd, coords)

        # Construct Partition of Unity

        self.cS = coarseSpace(self.da, self.A, self.comm, self.scatter_l2g, self.numSub, self.sub2proc, self.M)

        self.cS.buildPOU(True)

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
        elif((x[1] < 1e-6) or (x[2] < 1e-6) or (x[1] > L[1] - 1e-6) or (x[2] > L[2] - 1e-6)):
            type = 2 # Neumann Boundary
        else:
            type = 0 # Either internal or processor to processor boundary

        return val, type

    def solvePDE(self, plotSolution = False):

        # Solve A * x = b

        # Assemble Global Stifness Matrix

        b = self.da.createGlobalVec()
        b_local = self.da.createLocalVec()

        elem = self.da.getElements()
        nnodes = int(self.da.getCoordinatesLocal()[ :].size/self.dim)
        coords = np.transpose(self.da.getCoordinatesLocal()[:].reshape((nnodes,self.dim)))
        for ie, e in enumerate(elem,0): # Loop over all local elements
            Ke = self.fe.getLocalStiffness(coords[:,e], 1.0)
            self.A.setValuesLocal(e, e, Ke, PETSc.InsertMode.ADD_VALUES)
            b_local[e] = self.fe.getLoadVec(coords[:,e])
        self.A.assemble()
        self.comm.barrier()

        # Implement Boundary Conditions
        rows = []
        for i in range(nnodes):
            val, type = self.isBoundary(coords[:,i])
            if(type == 1): # It's Dirichlet
                rows.append(i)
                b_local[i] = val
        rows = np.asarray(rows,dtype=np.int32)
        self.A.zeroRowsLocal(rows, diag = 1.0)

        self.scatter_l2g(b_local, b, PETSc.InsertMode.INSERT_VALUES)

        # Solve

        # Setup linear system vectors
        x = self.da.createGlobalVec()
        x.set(0.0)

        # Setup Krylov solver - currently using AMG
        ksp = PETSc.KSP().create()
        pc = ksp.getPC()
        ksp.setType('cg')
        pc.setType('gamg')

        # Iteratively solve linear system of equations A*x=b
        ksp.setOperators(self.A)
        ksp.setInitialGuessNonzero(True)
        ksp.setFromOptions()
        ksp.solve(b, x)

        if(plotSolution): # Plot solution to vtk file
            viewer = PETSc.Viewer().createVTK('Solution.vts', 'w', comm = comm)
            x.view(viewer)
            viewer.destroy()

        return x

    def addtoBasis(self, x):
        # x is global soltuion vector from a previous solve. Need to obtain solution on subdomain
        x_loc = self.da.createLocalVec()
        self.scatter_l2g(x, x_loc, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        self.cS.addBasisElement(x_loc, comm.Get_rank())

    def buildCoarseSpace(self):

        self.cS.getCoarseVecs() # Builds Coarse Vectos on Each subdomain

        print(self.cS.needRebuild)

        self.cS.compute_Av()

L = [1.0, 1.0, 1.0]

n = [10, 10, 10]

dim = 3

overlap = 1

plotSolution = True

myModel = DarcyEx1(n, L, overlap, comm)

x = myModel.solvePDE(True)

myModel.addtoBasis(x) # Add solution to Coarse basis.

myModel.buildCoarseSpace()
