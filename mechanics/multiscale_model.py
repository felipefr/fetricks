#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 00:23:16 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np
from .material_model import materialModel

import fetricks as ft

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF

class multiscaleModel(materialModel):
    
    def __init__(self, W, Wtan, dxm, micromodels):
        
        self.micromodels = micromodels
        self.mesh = W.mesh()
        self.__createInternalVariables(W, Wtan, dxm)

    def __createInternalVariables(self, W, Wtan, dxm):
        self.stress = df.Function(W)
        self.eps = df.Function(W)
        self.tangent = df.Function(Wtan)
        
        self.size_tan = Wtan.num_sub_spaces()
        self.size_strain = W.num_sub_spaces()
    
        self.ngauss = int(W.dim()/self.size_strain)
        
        self.projector_eps = ft.LocalProjector(W, dxm, sol = self.eps)
        
        self.Wdofmap = W.dofmap()
        self.Wtandofmap = Wtan.dofmap()
        
        
    def tangent_op(self, de):
        return df.dot(ft.as_sym_tensor_3x3(self.tangent), de) 

    def update_stress_tangent(self):
        e = self.eps.vector().vec().array.reshape( (-1, self.size_strain) )
        s = self.stress.vector().vec().array.reshape( (-1, self.size_strain))
        t = self.tangent.vector().vec().array.reshape( (-1, self.size_tan))
        
        for i, m in enumerate(self.micromodels):
            s[i,:] , t[i,:] = m.getStressTangent_force(e[i,:])  
            
    def update(self, epsnew):
        self.projector_eps(epsnew) 
        self.update_stress_tangent()    

    