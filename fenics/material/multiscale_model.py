#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 00:23:16 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np
from .material_model import materialModel, materialModelExpression 

import fetricks as ft

as_sym_tensor = lambda a: df.as_tensor( [ [ a[0], a[1], a[2]] , [a[1] , a[3], a[4]] , [a[2] , a[4], a[5]] ])
ind_sym_tensor = np.array([0, 1, 2, 4, 5, 8])

collect_stress = lambda m, e: np.array( [ m[i].getStress(e[i,:]) for i in range(len(m))] ).flatten()
collect_tangent = lambda m, e: np.array( [ m[i].getTangent(e[i,:]).flatten()[ind_sym_tensor] for i in range(len(m))] ).flatten()

class multiscaleModel(materialModel):
    
    def __init__(self, W, Wtan, dxm, micromodels):
        
        self.micromodels = micromodels
        
        self.__createInternalVariables(W, Wtan, dxm)

    def __createInternalVariables(self, W, Wtan, dxm):
        self.stress = df.Function(W)
        self.eps = df.Function(W)
        self.tangent = df.Function(Wtan)
        
        self.eps.vector().set_local(np.zeros(W.dim()))
        
        self.num_cells = W.mesh().num_cells()
        
        self.projector_eps = ft.LocalProjector(W, dxm)
        
        self.size_tan = Wtan.num_sub_spaces()
        self.size_strain = W.num_sub_spaces()
    
            
    def tangent_op(self, de):
        return df.dot(as_sym_tensor(self.tangent), de) 

    def update(self, epsnew):
        
        self.projector_eps(epsnew ,  self.eps) 
        
        for m in self.micromodels:
            m.setUpdateFlag(False)
    
        strains = self.eps.vector().get_local()[:].reshape( (self.num_cells, self.size_strain) )
        
        self.stress.vector().set_local( collect_stress(self.micromodels, strains) ) 
        self.tangent.vector().set_local( collect_tangent(self.micromodels, strains) ) 
        

class multiscaleModelExpression(materialModelExpression):
    
    def __init__(self, W, Wtan, dxm, micromodels):
        self.micromodels = micromodels
        super().__init__(W, Wtan, dxm)
    
    def pointwise_stress(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format        
        return self.micromodels[cell.index].getStress(e)
    
    def pointwise_tangent(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        return self.micromodels[cell.index].getTangent(e).flatten()[ind_sym_tensor]
    
    def tangent_op(self, de):
        return df.dot(as_sym_tensor(self.tangent), de) 
    
    def update(self, e):
        super().update(e)
        
        for m in self.micromodels:
            m.setUpdateFlag(False)
    