#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:22:08 2023

@author: ffiguere


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
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
