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

def isDirichlet(x, L):
    output = False
    val = 0.0
    if(x[0] < 1e-6):
        output = True
        val = 0.0
    if(x[0] > L[0] - 1e-6):
        output = True
        val = 1.0
    return output, val

L = [2.0, 1.0, 1.0]

n = [20, 10, 10]

dim = 3

overlap = 1

plotSolution = True

da = PETSc.DMDA().create(n, dof=1, stencil_width=overlap)
da.setUniformCoordinates(xmax=L[0], ymax=L[1], zmax=L[2])
da.setMatType(PETSc.Mat.Type.AIJ)

# Define Finite Element space

fe = DarcyQ1(dim)

# Setup global and local matrices + communicators

A = da.createMatrix()
r, _ = A.getLGMap() # Get local to global mapping
is_A = PETSc.IS().createGeneral(r.indices) # Create Index Set for local indices
A_local = A.createSubMatrices(is_A)[0] # Construct local submatrix on domain
vglobal = da.createGlobalVec()
vlocal = da.createLocalVec()
scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, is_A)

# Assemble Global Stifness Matrix

b = da.createGlobalVec()
b_local = da.createLocalVec()

elem = da.getElements()
nnodes = int(da.getCoordinatesLocal()[ :].size/dim)
coords = np.transpose(da.getCoordinatesLocal()[:].reshape((nnodes,dim)))
for ie, e in enumerate(elem,0): # Loop over all local elements
    Ke = fe.getLocalStiffness(coords[:,e], 1.0)
    A.setValuesLocal(e, e, Ke, PETSc.InsertMode.ADD_VALUES)
    b_local[e] = fe.getLoadVec(coords[:,e])
A.assemble()
comm.barrier()

# Implement Boundary Conditions
rows = []
for i in range(nnodes):
    isDirich, val = isDirichlet(coords[:,i], L)
    if(isDirich):
        rows.append(i)
        b_local[i] = val
rows = np.asarray(rows,dtype=np.int32)
A.zeroRowsLocal(rows)

scatter_l2g(b_local, b, PETSc.InsertMode.INSERT_VALUES)

# Solve

# Setup linear system vectors
x = da.createGlobalVec()
x.set(0.0)

# Setup Krylov solver - currently using AMG
ksp = PETSc.KSP().create()
pc = ksp.getPC()
ksp.setType('cg')
pc.setType('gamg')

# Iteratively solve linear system of equations A*x=b
ksp.setOperators(A)
ksp.setInitialGuessNonzero(True)
ksp.setFromOptions()
ksp.solve(b, x)

if(plotSolution): # Plot solution to vtk file
    viewer = PETSc.Viewer().createVTK('Solution.vts', 'w', comm = comm)
    x.view(viewer)
    viewer.destroy()
