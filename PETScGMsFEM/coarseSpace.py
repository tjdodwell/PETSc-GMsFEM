
from petsc4py import PETSc
import mpi4py.MPI as mpi
import numpy as np
from slepc4py import SLEPc


class coarseSpace():

    # Contains and builds Coarse Space for Multiscale Method and Preconditioner

    def __init__(self, da, A, comm, scatter_l2g):

        self.da = da

        self.A = A

        self.localBasis = []

        self.comm = comm

        self.scatter_l2g = scatter_l2g

        # Setup global and local vectors and matrices

        r, _ = self.A.getLGMap()
        self.is_A = PETSc.IS().createGeneral(r.indices)
        self.A_local = self.A.createSubMatrices(self.is_A)[0] # Construct local submatrix on domain

    def buildPOU(self, plotPOU = False):

        # Build Partition of Unity Operator - Compute the multiplicity of each degree of freedom and max of the multiplicity
        xi_local,_ = self.A_local.getVecs()
        xi_global,_ = self.A.getVecs()
        xi_local.set(1.)
        xi_global.set(0.)
        self.scatter_l2g(xi_local, xi_global, PETSc.InsertMode.ADD_VALUES)
        self.scatter_l2g(xi_global, xi_local, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
        for i in range(xi_local[:].size):
            xi_local[i] = 1.0 / xi_local[i]
        self.xi_local = xi_local
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

"""
        # Add constant function for those domains without Dirichlet Boundary
        if(self.grid.Dirich.size == 0): # No nodes on Dirichlet boundaries
            zeroEnergy,_ = A_local.getVecs()
            zeroEnergy.set(1.0)
            self.localBasis.append(zeroEnergy)



    def project(self,u,v):
        tmp = u.copy()
        tmp.scale(v.dot(u) / u.dot(u))
        return tmp

    def GramSchmidt(self):
        # Applies GramSchmidt to self.local_basis
        size_local_basis = self.local_basis
        newBasis = []
        for i in range(size_local_basis):
            newVec = self.local_basis[i].copy()
            for j in range(0,i):
                tmpVec = self.local_basis[i].copy()
                newVec -= self.project(tmp,newBasis[j])
            newBasis.append(newVec)
        # Normalise new basis in L2
        for i in range(newBasis.len()):
            newBasis[i].scale(1./np.sqrt(newBasis[i].dot(newBasis[i])))
        return newBasis

    def addBasisVector(self, x):
        self.local_basis.append(x.copy())
        self.size_local_basis += 1
        self.GramSchmidt()


    def build_AH(self, A):


        build_AH() - builds a coarse matrix for full matrix A using self.local_basis

        returns AH - Coarse Matrix + PETSc Solver kspAH for problem

        - Implementation differs from petsc-geneo - Need to understand why.

        - Not A can be any matrix

        - Where do we implement partition of unity - probably in self.local_basis

        - How do we implement boundary conditions.



        # Communicate sizes of local basis
        coarse_vecs = [] # List containing all coarse modes
        for i in range(mpi.COMM_WORLD.size): # For each processor
            size_local_basis = self.local_basis.len() if i == mpi.COMM_WORLD.rank else None
            size_local_basis = mpi.COMM_WORLD.bcast(size_local_basis, root=i)
            for j in range(size_local_basis):
                coarse_vecs.append(self.local_basis[j] if i == mpi.COMM_WORLD.rank else None)

        # Construct A * v for each of the local basis functions

        coarse_Avecs = [] # Initialise List
        work, _ = self.A.getVecs()
        workl, _ = self.A_local.getVecs()
        for vec in coarse_vecs: # For each of the vectors in coarse space
            if vec: # if this vec belongs to this processor
                workl = X * vec.copy()
            else:
                workl.set(0.0) # else set to zero
            work.set(0.0) # set global v
            self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)
            coarse_Avecs.append(A * work) # A * v for all basis functions
            self.scatter_l2g(coarse_Avecs[-1], workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

        # Build Coarse Space Matrix

        # Define, fill and factorize coarse problem matrix
        AH = PETSc.Mat().create(comm=PETSc.COMM_SELF) # Create Coarse Matrix System Locally on each processor
        AH.setType(PETSc.Mat.Type.SEQDENSE)
        AH.setSizes([len(coarse_vecs),len(coarse_vecs)])
        AH.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        AH.setPreallocationDense(None)

        for i, vec in enumerate(coarse_vecs):

            if vec: # if this vec belongs to this processor
                workl = X * vec.copy()
            else:
                workl.set(0.0) # else set to zero

            work.set(0)
            self.scatter_l2g(self.workl, work, PETSc.InsertMode.ADD_VALUES)

            for j in range(i+1): # Going to loop over just the lower triangle
                tmp = coarse_Avecs[j].dot(work) # AH[i,j] = v_j^T A v_i
                AH[i, j] = tmp
                AH[j, i] = tmp

        AH.assemble() # Assemble coarse Matrix

        # Set up coarse space solver

        ksp_AH = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp_AH.setOperators(AH)
        ksp_AH.setType('preonly')
        pc = ksp_AH.getPC()
        pc.setType('cholesky')

        return AH, ksp_AH
"""
