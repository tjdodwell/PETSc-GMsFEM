import numpy as np

from .HEX import HEX

class CohesiveQ1(HEX):

    def __init__(self, dim = 3, order_integration = "full"):

        super().__init__(dim, 1, order_integration)

        self.dofel = 24

    def getLocalStiffness(self, x, k):
        # getKe - computes element stiffness matrix
        # x - coordinates of element nodes np.array - dim by nodel
        # val -  permeability value at each node
        Ke = np.zeros((self.dofel,self.dofel))

        k_ip = k#   np.mean(k)
        for ip in range(self.nip): # For each integration point
            J = np.matmul(x, self.dNdu[ip])
            dNdX = np.matmul(self.dNdu[ip],np.linalg.inv(J))

            B = []

            np.matmul(np.transpose(B), np.matmul(C, B))

            Ke += k_ip * np.matmul(dNdX,np.transpose(dNdX)) * np.linalg.det(J) * self.IP_W[ip]
        return Ke


    def getLocalMass(self, x, val = 1.0):
        # getMe - computes element mass matrix
        # x - coordinates of element nodes np.array - dim by nodel
        # val - scalar permeability value for element
        Me = np.zeros((self.dofel,self.dofel))
        for ip in range(self.nip): # For each integration point
            J = np.matmul(x, self.dNdu[ip])
            Me += val * np.matmul(self.N[ip],np.transpose(self.N[ip])) * np.linalg.det(J) * self.IP_W[ip]
        return Ke

    def getLoadVec(self, x, func = 1.0):
        # getRHS - computes load vector
        # x - coordinates of element nodes np.array - dim by nodel
        # func - function handle to evaluate source function at given x in Omega
        fe = np.zeros(8)
        for ip in range(self.nip):
            J = np.matmul(x, self.dNdu[ip])
            x_ip = np.matmul(x,self.N[ip])
            val = func#(x_ip)
            fe = fe + val * self.N[ip] * np.linalg.det(J) * self.IP_W[ip]

        return fe
