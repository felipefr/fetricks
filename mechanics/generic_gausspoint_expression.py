#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:49:07 2022

@author: felipe


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import dolfin as df

class genericGaussPointExpression(df.UserExpression):
    def __init__(self, strain, pointwiseLaw, shape,  **kwargs):
        self.strain = strain
        self.pointwiseLaw = pointwiseLaw
        self.shape = shape
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        strain = self.strain.vector().get_local()[cell.index*3:(cell.index + 1)*3]
        values[:] = self.pointwiseLaw(strain, cell).flatten()

    def value_shape(self):
        return self.shape