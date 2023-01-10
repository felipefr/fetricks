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
import pygmsh


code_gmsh = """
p12 = newp;
Point(p12) = {0.0, 0.0, 0.0};
p13 = newp;
Point(p13) = {20.0, 0.0, 0.0};
p14 = newp;
Point(p14) = {20.0, 50.0, 0.0};
p15 = newp;
Point(p15) = {0.0, 50.0, 0.0};
l12 = newl;
Line(l12) = {p12, p13};
l13 = newl;
Line(l13) = {p13, p14};
l14 = newl;
Line(l14) = {p14, p15};
l15 = newl;
Line(l15) = {p15, p12};
ll3 = newll;
Line Loop(ll3) = {l12, l13, l14, l15};
s3 = news;
Plane Surface(s3) = {ll3};
Transfinite Line {l12} = 5;
Transfinite Line {l13} = 10;
Transfinite Line {l14} = 5;
Transfinite Line {l15} = 10;
Physical Surface(0) = {s3};
Physical Line(1) = {l12};
Physical Line(2) = {l14};
"""

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
    
    
    # From raw code Gmsh (non parametrised, but easily done)
    gmsh_mesh = ft.Gmsh("bar.xdmf")
    gmsh_mesh.add_raw_code(code_gmsh)

    gmsh_mesh.generate()
    #gmsh_mesh.writeMSH() # automatically generate mesh object
    gmsh_mesh.write("fenics", optimize_storage = True)

    mesh = ft.Mesh("bar.xdmf")
    
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(1)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(2)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.dx(0)), l*h)


    # From the class
    gmsh_mesh = Bar("bar.xdmf", h, l, Ntransfinite = [Nbottom, Nright, Ntop, Nleft])
    gmsh_mesh.generate()
    # gmsh_mesh.writeMSH() # automatically generate mesh object
    gmsh_mesh.write("fenics", optimize_storage = False)

    mesh = ft.Mesh("bar.xdmf")
    
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(1)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(2)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.dx(0)), l*h)

    
    # From the class
    gmsh_mesh = Bar("bar.xdmf", h, l, Ntransfinite = [Nbottom, Nright, Ntop, Nleft])
    gmsh_mesh.generate()
    # gmsh_mesh.writeMSH() # automatically generate mesh object
    gmsh_mesh.write("fenics", optimize_storage = True)

    mesh = ft.Mesh("bar.xdmf")
    
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(1)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(2)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.dx(0)), l*h)
    
    
    
    

