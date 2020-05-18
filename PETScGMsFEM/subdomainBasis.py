from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

import time


class subdomainBasis():

    # subdomainBasis() - base class for local subdomain

    def __init__(self, da, comm):

        self.da = da

        self.comm = comm

        self.rank = comm.Get_rank()

        self.basis = []

        self.globalID = []

        self.isBnd = []

        self.bndDof = []

        self.Aphi_i = []

        self.totalSize = 0


        # Setup global and local matrices + communicators
        self.A = da.createMatrix()
        r, _ = self.A.getLGMap() # Get local to global mapping
        is_A = PETSc.IS().createGeneral(r.indices) # Create Index Set for local indices
        self.A_local = self.A.createSubMatrices(is_A)[0] # Construct local submatrix on domain
        vglobal = self.da.createGlobalVec()
        vlocal = self.da.createLocalVec()
        self.scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, is_A)

        self.nnode_local = vlocal[:].size

        # Build POU

        self.buildPOU(True)

        # Build Nearest neighbours

        self.getNearestNeighbours()

        self.communicationList()

        self.numNeighbours = self.neighbour_list[self.rank].size


    def __getitem__(self, key):

        return self.basis[key]

    def clear(self, bnd = False):
        tmp = self.basis[0].copy()
        self.basis.clear()
        self.isBnd.clear()
        self.globalID.clear()
        self.bndDof.clear()
        if(bnd == False):
            self.add(tmp,isBnd = True)

    def overwrite(self, vecs, bnd = False, POD = True):

        if(POD == False):
            self.clear(bnd)
            for i, vec in enumerate(vecs):
                self.add(vec, isBnd = False)
        else:
            self.basis.clear()
            self.isBnd.clear()
            self.globalID.clear()
            self.bndDof.clear()

            for i, vec in enumerate(vecs):
                if(i == 0):
                    self.add(vec, isBnd = True)
                else:
                    self.add(vec, isBnd = False)


    def GramSchmidt(self, Test = True):

        k = self.size()

        print("Size of local basis = " + str(k))

        # First one is left for the boundary condition

        tmp = self.basis.copy()

        self.clear() # Clears basis

        print(self.basis)

        norm2 = tmp[1].norm(PETSc.NormType.NORM_2)

        self.add(tmp[1].copy())

        self.basis[1].scale(1./ norm2)

        for i in range(2, k):

            print("This is " + str(i))

            self.add(tmp[i].copy())

            print(self.basis)

            for j in range(1, i):
                c1 = self.basis[j].dot(self.basis[i])
                c2 = self.basis[j].dot(self.basis[j])

                self.basis[i][:] -= (c1/c2) * self.basis[j][:]

            self.basis[i][:] /= self.basis[i].norm(PETSc.NormType.NORM_2)

        if(Test):


            ans = self.basis[1].dot(self.basis[2])

            print(ans)
            assert np.abs(ans) < 1e-4, "Gram-Schmidt not working"

        for i in range(self.size()):
            self.basis[i] = np.dot(self.X,self.basis[i])


    def add(self, vec, id_global = None, isBnd = False):

        self.basis.append(vec)
        self.globalID.append(id_global)
        self.isBnd.append(isBnd)

    def size(self):
        return len(self.basis)

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

    def getGlobalIDs(self):

        lsize = np.zeros(1, dtype = np.int32)

        lsize[0] = self.size()

        self.localSizes = np.zeros(self.comm.Get_size(), dtype = np.int32)

        self.comm.Allgather([lsize, mpi.INT], [self.localSizes, mpi.INT])

        self.totalSize = np.sum(self.localSizes)

        start = np.sum(self.localSizes[0:self.comm.Get_rank()])

        for j in range(self.localSizes[self.comm.Get_rank()]):
            self.setGlobalID(j, start + j)
            if(self.isBnd[j]):
                self.bndDof.append(start + j)

    def setGlobalID(self, id_local, id_global):
        self.globalID[id_local] = id_global

    def getNearestNeighbours(self):

        self.neighbour_list = []

        self.overlap_dofs_send = [ ]
        self.overlap_dofs_recv = [ ]

        # dof_proc - list of dofs belong to proc which will shared with other processors

        self.dof_proc = [ [] for i in range(self.comm.Get_size())]

        # dof_other - list of dofs belong to proc which will shared with other processors

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
                # This is a nearest neigbour
                isNeighbour[0] = 1
                id = np.asarray(np.nonzero(workl[:]), dtype = np.int32)
                #self.dof_other[i].append(id) # Record local nodes


            self.comm.Allgather([isNeighbour, mpi.INT], [Neighbours, mpi.INT])

            self.neighbour_list.append(np.nonzero(Neighbours)[0])

            # All Processors know about global connectivity

            for k in self.neighbour_list[-1]: # For the neighbours of processor i

                if(self.comm.Get_rank() == k): # If you are the neighbour
                    workl.set(1.0)
                else:
                    workl.set(0.0)

                work.set(0.0)

                self.scatter_l2g(workl, work, PETSc.InsertMode.ADD_VALUES)

                workl.set(0.0)
                self.scatter_l2g(work, workl, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)

                if(self.comm.Get_rank() == i): # If this processor
                    id = np.asarray(np.nonzero(workl[:]), dtype = np.int32)
                    self.dof_proc[k].append(id) # Record local nodes which will communicate with neighbour k

            #if(self.rank == i):
            #    print(self.dof_proc)


    def communicationList(self):
        self.comm_list = []
        for i, vec in enumerate(self.neighbour_list[self.rank]):
            if(self.rank < vec):
                self.comm_list.append([self.rank, vec])
            else:
                self.comm_list.append([vec, self.rank])


    def build_Nearest_Neighbour_Basis(self):

        # Build Nearest Neighbour Communication for New Basis

        data_recv = [ [] for i in range(self.comm.Get_size())]

        self.neighbour_basis = [ [] for i in range(self.comm.Get_size())]

        self.neighbour_basis_ids = [ [] for i in range(self.comm.Get_size())]

        for i, vec in enumerate(self.comm_list):

            other = 0
            this = 1
            if(vec[0] == self.rank):
                other = 1
                this = 0

            id = self.dof_proc[vec[other]][0][0]

            # Sending to other processor
            for j in range(self.size()): # For each local basis

                data_send = self.basis[j][self.dof_proc[vec[other]][0][0]]

                print(self.dof_proc)

                self.comm.Isend(data_send, dest=vec[other])#, tag = self.generateTag(vec[this], vec[other], j))

            # Receive from other processor

            start = np.sum(self.localSizes[0:vec[other]])

            for j in range(self.localSizes[vec[other]]): # For all the basis functions on the other subdaomin

                data_tmp = np.empty(len(id), dtype=np.float64)

                req = self.comm.Irecv(data_tmp, source=vec[other])#, tag = self.generateTag(vec[other], vec[this], j))

                req.Wait()

                data_recv[vec[other]].append(data_tmp)

                # Store data locally on processor

                self.neighbour_basis[vec[other]].append(self.da.createLocalVec())

                self.neighbour_basis[vec[other]][-1].set(0.0)

                self.neighbour_basis[vec[other]][-1].setValues(id, data_tmp)

                self.neighbour_basis_ids[vec[other]].append(start + j)



        self.comm.barrier()
