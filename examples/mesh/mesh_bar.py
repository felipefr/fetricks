#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:22:08 2023

@author: ffiguere
"""


import numpy as np

import fetricks as ft
import numpy as np
import dolfin as df
import meshio 

if __name__ == '__main__':
    
    h = 50.0;
    l = 20.0;
    
    # generation from .msh : msh -> xdmf or xml
    model = ft.GmshIO('bar_generated_from_gmsh.msh', 2)
    model.write(option = 'xdmf')

    # test
    mesh = ft.Mesh("bar_generated_from_gmsh.xdmf")
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(1)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(2)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.dx(0)), l*h)

    # generation from .geo: geo -> msh -> xdmf or xml
    model = ft.GmshIO('bar.geo', 2)
    model.write(option = 'xdmf')

    # test    
    mesh = ft.Mesh("bar.xdmf")    
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(1)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(2)), l)
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.dx(0)), l*h)