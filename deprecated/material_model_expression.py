#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:48:27 2022

@author: felipe

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import sys
import dolfin as df
import numpy as np

from .generic_gausspoint_expression import genericGaussPointExpression
import fetricks as ft

class materialModelExpression:
    def __init__(self, W, dxm):
        self.strain = df.Function(W) 
        self.projector = ft.LocalProjector(W, dxm)
        
        self.stress = genericGaussPointExpression(self.strain, self.stress_op , (3,))
        self.tangent = genericGaussPointExpression(self.strain, self.tangent_op , (3,3,))
        
    def stress_op(self, e, cell = None):
        pass
    
    def tangent_op(self,e, cell = None):
        pass
    
    def update(self, e):
        self.projector(e, self.strain)
