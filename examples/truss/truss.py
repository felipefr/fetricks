# Example truss

import os, sys
os.environ['Hufl5_DISABLE_VERSION_CHECK']='2'
sys.path.append("/home/felipe/sources/fetricksx")

import numpy as np
from dolfinx import fem, io
from mpi4py import MPI

import fetricksx as ft

if __name__ == '__main__':
    
    L = 3.6 
    X = np.array([[0., 0.],
                    [L, 0.],
                    [2*L, 0.],
                    [0., L],
                    [L, L],
                    [2*L, L]])

    cells = np.array([  [0, 1],
                        [1, 2],
                        [3, 4],
                        [4, 5],
                        [2, 5],
                        [1, 4],
                        [1, 5],
                        [2, 4],
                        [0, 4],
                        [1, 3]], dtype = int)
    
    
    fy = -1.0      
    E = 100.0
    

# for physical groups: tuple of (tdim, node, tag)
# for dirichlet and neumann: tuple of (physical group tag, direction, value)
    param = {
    'physical_groups': [(0,0,1), (0,3,2), (0,1,3), (0,2,4) ],
    'dirichlet': [(1, 0, 0. ), (1, 1, 0. ),
                  (2, 0, 0. ), (2, 1, 0. )],
    'neumann': [(3, 1 , fy), (4, 1 , fy)], 
    'X': X,
    'cells': cells
    }
    
    domain, markers, facets = ft.get_mesh_truss(X, cells, param)
    mesh = (domain, markers, facets)
    param['sigma_law'] =  lambda e: fem.Constant(domain, E)*e
    param['Area'] = fem.Constant(domain, 1.0)

    uh = ft.solve_truss(param, mesh)
    zh = ft.posproc_truss(param, uh, domain)
    
    with io.XDMFFile(MPI.COMM_WORLD, "truss.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
        xdmf.write_function(zh["stress"])
        xdmf.write_function(zh["strain"])
            


