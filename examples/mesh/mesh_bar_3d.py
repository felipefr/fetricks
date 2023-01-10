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
    
    l = 50.0; # x
    h = 20.0; # y
    t = 10.0; # z
    
    # generation from .geo: geo -> msh -> xdmf or xml
    model = ft.GmshIO('bar3d.geo', 3)
    model.write(option = 'xdmf')

    # test    
    mesh = ft.Mesh("bar3d.xdmf")    
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(1)), h*t) # left
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(2)), h*t) # right
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(3)), l*h) # bottom
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(4)), l*h) # top
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(5)), l*t) # back
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.ds(6)), l*t) # front
    assert np.allclose(df.assemble( df.Constant(1.0)*mesh.dx(0)), l*h*t)
