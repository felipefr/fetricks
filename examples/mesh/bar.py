#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:53:28 2023

@author: ffiguere
"""

import numpy as np

import fetricks as ft
import numpy as np
import dolfin as df


class Bar(ft.Gmsh):
    def __init__(self, meshname, h, l, Ntransfinite = []):
        super().__init__(meshname)    
        
        self.rec = self.add_rectangle(0.0, l, 0.0, h, 0.0, make_surface= False) 
        
        self.s = self.add_plane_surface(self.rec.line_loop)
        
        for i, N in enumerate(Ntransfinite):
            self.set_transfinite_lines([self.rec.lines[i]], N)
        
        self.physicalNaming()
        
    
    def physicalNaming(self):
        self.add_physical(self.s, 0)
        self.add_physical(self.rec.lines[0], 1) # bottom
        self.add_physical(self.rec.lines[2], 2) # top
        
if __name__ == '__main__':
    h = 50.0;
    l = 20.0;
    r = 2.0;

    Nbottom = 5
    Ntop = 5
    Nleft = 10
    Nright = 10


    gmsh_mesh = Bar("bar.xdmf", h, l, Ntransfinite = [Nbottom, Nright, Ntop, Nleft])
    gmsh_mesh.generate()
    # gmsh_mesh.writeMSH() # automatically generate mesh object
    gmsh_mesh.write("fenics", optimize_storage = False)

    mesh = ft.Mesh("bar.xdmf")
    
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(1)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(2)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.dx(0)), l*h)

    

    gmsh_mesh = Bar("bar.xdmf", h, l, Ntransfinite = [Nbottom, Nright, Ntop, Nleft])
    gmsh_mesh.generate()
    # gmsh_mesh.writeMSH() # automatically generate mesh object
    gmsh_mesh.write("fenics", optimize_storage = True)

    mesh = ft.Mesh("bar.xdmf")
    
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(1)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(2)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.dx(0)), l*h)
