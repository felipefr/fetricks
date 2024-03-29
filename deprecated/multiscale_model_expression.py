#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:50:48 2022

@author: felipe

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""
import dolfin as df
import numpy as np

from .material_model_expression import materialModelExpression
from micmacsfenics.core.fenicsUtils import (symgrad, tensor2mandel,  mandel2tensor, tr_mandel, Id_mandel_np)

class multiscaleModelExpression(materialModelExpression):
    
    def __init__(self, W, dxm, micromodels):
        self.micromodels = micromodels
        super().__init__(W, dxm)
    
    def stressHomogenisation(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format        
        return self.micromodels[cell.index].getStress(e)
    
    def tangentHomogenisation(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        
        return self.micromodels[cell.index].getTangent(e)
    
    def update(self, e):
        super().update(e)
        
        for m in self.micromodels:
            m.setUpdateFlag(False)
    