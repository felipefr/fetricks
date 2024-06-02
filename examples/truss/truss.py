# Example truss

import os, sys
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
sys.path.append("/home/felipe/sources/fetricks")

import dolfin as df 
import numpy as np

import fetricks as ft



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
    p1 = df.Point((0.0, 0.0)) # node 0
    p2 = df.Point((0, L)) # node 3           
    p3 = df.Point((L, 0.0)) # node 1
    p4 = df.Point((2*L, 0.0)) # node 2           
    E = 100.0
    

# for dirichlet and neumann: tuple of (node (int or df.Point), direction, value)
    param = {
    'dirichlet': [(p1, 0, 0. ), (p1, 1, 0. ),
                  (p2, 0, 0. ), (p2, 1, 0. )],
    'neumann': [(p3, 1 , fy), (p4, 1 , fy)], 
    'sigma_law': lambda e: df.Constant(E)*e,
    'Area': df.Constant(1.0), 
    'X': X,
    'cells': cells
    }
    
    
    uh = ft.solve_truss(param)
    zh = ft.posproc_truss(param, uh)
    
    uh.rename('u', '')
    
    ft.exportXDMF_gen("truss_vtk_2.xdmf", fields={'vertex': [uh], 'cell': [zh['strain'], zh['stress']]})
    


