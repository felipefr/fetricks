#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:57:46 2021

@author: felipefr


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>


Example of usage:
Given the rectangle limits: x0, x1, y0, y1 and the mesh
 
periodicity = PeriodicBoundary(x0, x1, y0, y1])
V = df.VectorFunctionSpace(mesh, "CG", polyorder, constrained_domain=periodicity)
"""

import dolfin as df

class PeriodicBoundary(df.SubDomain):
    # Left boundary is "target domain" G
    def __init__(self, x0=0.0, x1=1.0, y0=0.0, y1=1.0, **kwargs):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT
        # on one of the two corners (0, 1) and (1, 0)
        if(on_boundary):
            left, bottom, right, top = self.checkPosition(x)
            return (left and not top) or (bottom and not right)

        return False

    def checkPosition(self, x):
        
        return [df.near(x[0], self.x0), df.near(x[1], self.y0),
                df.near(x[0], self.x1), df.near(x[1], self.y1)]

    def map(self, x, y):
        left, bottom, right, top = self.checkPosition(x)

        y[0] = x[0] + self.x0 - (self.x1 if right else self.x0)
        y[1] = x[1] + self.y0 - (self.y1 if top else self.y0)


