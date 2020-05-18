
import numpy as np
import time

from numba import jit

def bubbleSort(lam):
    N = lam.size
    index = np.arange(N)
    sw = 1
    while(sw == 1):
        sw = 0
        for i in range(N-1):
            if(lam[i] < lam[i+1]):
                tmpreal = lam[i+1]
                tmpint = index[i+1]
                lam[i+1] = lam[i]
                index[i+1] = index[i]
                lam[i] = tmpreal
                index[i] = tmpint
                sw = 1
    return lam, index

@jit(nopython=True)
def evalPhi(x, L, ell, omega):
    x /= L
    l = ell / L
    norm = np.sin(2.0*omega)*(0.25*(l*l*omega - 1.0/omega)) - 0.5 * l * np.cos(2.0 * omega) + 0.5 * ( 1 + l + l*l*omega*omega);
    norm = 1.0 / np.sqrt(norm);
    phi = norm * (np.sin(x * omega) + l * omega * np.cos(x * omega) );
    return phi



class SquaredExp:

    # Class calculates the random field associated with Square Exponential Kernel

    def __init__(self, comm, dim = 3, L = 1.0, ell = 0.3, mkl = 125, sigKL = 1.0, verbose = False):

        self.comm = comm

        self.dim = dim; # Dimension of physical space
        self.mkl = mkl  # self.mkl - Number of KL modes
        self.oneD_mkl = int(np.ceil(mkl**(1./self.dim)))
        self.ell = ell
        self.L = L
        self.sigKL = sigKL

        self.verbose = verbose

    def constructTensorProduct(self):
        N = int(self.oneD_mkl ** self.dim)
        index = np.zeros((self.dim, N), dtype = int)
        self.index = np.zeros((N, self.dim), dtype = int)
        self.lam = np.zeros(N)
        ind = np.zeros(N)
        count = 0
        if(self.dim == 2):
            for i in range(0,self.oneD_mkl):
                for j in range(0, self.oneD_mkl):
                    self.lam[count] = self.lambda_oneD[i] * self.lambda_oneD[j]
                    ind[count] = count;
                    index[0][count] = i
                    index[1][count] = j
                    count += 1
        elif(self.dim == 3):
            for i in range(0,self.oneD_mkl):
                for j in range(0, self.oneD_mkl):
                    for k in range(0, self.oneD_mkl):
                        self.lam[count] = self.lambda_oneD[i] * self.lambda_oneD[j] * self.lambda_oneD[k]
                        ind[count] = count;
                        index[0][count] = i
                        index[1][count] = j
                        index[2][count] = k
                        count += 1
        else:
            self.lam = self.lambda_oneD
            for i in range(0, self.oneD_mkl):
                index[0][count] = i

        self.lam, ind = bubbleSort(self.lam)

        for i in range(0, N):
            self.index[i][0] = index[0][ind[i]]
            if(self.dim > 1):
                self.index[i][1] = index[1][ind[i]]
                if(self.dim > 2):
                    self.index[i][2] = index[2][ind[i]]

    def construct(self):

        self.freq_oneD = self.rootFinder(self.oneD_mkl, self.ell / self.L)
        self.lambda_oneD = self.evaluate1Deigenvalues(self.ell / self.L, self.freq_oneD)

        self.constructTensorProduct()

    def loadRandomField(self, filename="RandomField"):

        self.RF = np.load(filename + "_" + str(self.comm.Get_rank()) + ".npy")


    def saveRandomField(self, filename="RandomField"):

        np.save(filename + "_" + str(self.comm.Get_rank()), self.RF)

    def computeRandomField(self, da):
        dim = 3
        elem = da.getElements()
        nnodes = int(da.getCoordinatesLocal()[ :].size/dim)
        coords = da.getCoordinatesLocal()[:].reshape((nnodes,dim))
        nel = elem.shape[0]
        self.RF = np.zeros((nnodes, self.mkl))
        for i in range(nnodes):
            for j in range(self.mkl):
                self.RF[i, j] = np.sqrt(self.lam[j]) * evalPhi(coords[i,0], self.L, self.ell, self.freq_oneD[self.index[j][0]])  * evalPhi(coords[i,1], self.L, self.ell, self.freq_oneD[self.index[j][1]]) * evalPhi(coords[i,2], self.L, self.ell, self.freq_oneD[self.index[j][2]])

    def evaluate1Deigenvalues(self, ell, freq):
        c = 1 / ell
        return 2.0 * c / (freq**2 + c**2)

    def res(self,x,ell):
        c = 1 / ell
        g = np.tan(x) - 2.0 * c * x / (x * x - c * c)
        return g

    def findRoot(self,ell, a, b):
        # Regla-Falsi Method
        error = 1.0
        while(error > 1e-6):
            fa = self.res(a, ell)
            fb = self.res(b,ell)
            m = (fb - fa) / (b - a)
            x = a - fa / m
            fx = self.res(x,ell)
            if (((fa < 0) & (fx < 0)) | ((fa > 0) & (fx > 0))):
                a = x
            else:
                b = x;
            error = np.abs(fx);
        return x

    def rootFinder(self, M, ell):
        c = 1 / ell
        freq = np.zeros(M + 2)
        m = -1
        for i in range(0, M + 1):
            w_min = (i - 0.4999) * np.pi
            w_max = (i + 0.4999) * np.pi
            if ((w_min <= c) and (w_max >= c)):
                # If not first interval look for solution near left boundary
                if (w_min > 0.0):
                    m += 1
                    freq[m] = self.findRoot(ell,w_min,0.5*(c+w_min));
                # Always look for solution near right boundary
                m += 1;
                freq[m] = self.findRoot(ell,0.5*(c + w_max),w_max);
            else:
                m += 1;
                freq[m] = self.findRoot(ell,w_min,w_max)
        return freq[1:M+1]
