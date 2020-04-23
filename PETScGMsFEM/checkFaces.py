import numpy as np

def checkFaces(da, isBnd, coords_local):
    # Loops over all faces of the boundary of a domain - marking dirichlet, neumann and processor to processor boundarie
    ranges = da.getGhostRanges()
    dim = da.getDim()
    sizes = np.empty(dim, dtype=np.int32)
    for ir, r in enumerate(ranges):
        sizes[ir] = r[1] - r[0]
    loops = [[2, 1], [2, 1], [2,0], [2,0], [1,0], [1,0]]
    start = [[0, -1, -1], [sizes[0]-1, -1, -1], [-1, 0, -1], [-1, sizes[1]-1, -1], [-1, -1, 0], [-1, -1, sizes[2]-1]]
    tmpDirich = []
    tmpNeumann = []
    tmpP2P = []
    for iFace in range(6):
        lC = start[iFace]
        for k in range(sizes[loops[iFace][0]]):
            lC[loops[iFace][0]] += 1
            lC[loops[iFace][1]] = -1
            for j in range(sizes[loops[iFace][1]]):
                lC[loops[iFace][1]] += 1
                id = lC[0] + sizes[0] * lC[1] + sizes[0] * sizes[1] * lC[2]
                _,type = isBnd(coords_local[:,id])
                if(type == 1): # Dirichlet
                    tmpDirich.append(id)
                elif(type == 2): # Neumann
                    tmpNeumann.append(id)
                else:
                    tmpP2P.append(id)

    Dirich = np.unique(np.array(tmpDirich))
    Neumann = np.unique(np.array(tmpNeumann))
    P2P = np.unique(np.array(tmpP2P))

    return Dirich, Neumann, P2P
