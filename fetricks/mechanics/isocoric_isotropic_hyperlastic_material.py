#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:55:13 2023

@author: ffiguere
"""

import dolfin as df
import matplotlib.pyplot as plt
import ufl
import numpy as np
import fetricks as ft 

from timeit import default_timer as timer
from functools import partial
        
class IsochoricIsotropicHyperelasticMaterial(ft.materialModel):
            
    def param_parser(self, param):
        
        self.gdim = param['gdim']
        self.conv = ft.get_mechanical_notation_conversor(gdim = self.gdim)
            
    def create_internal_variables(self):
        self.stress = df.Function(self.W)
        self.strain = df.Function(self.W)
        self.tangent = df.Function(self.Wtan)
        
        projector_strain = ft.LocalProjector(self.W, self.dxm, self.strain)
        self.projector_list = {'strain' : projector_strain}

    def stress_op(self, E):
        return self.stress
        
    def tangent_op(self, de):
        return df.dot(self.tangent, de)
    
    def update(self, u):
        alpha_new = {'strain' : ft.get_GL_mandel(u)}
        self.project_var(alpha_new)
        
        strain_table = self.strain.vector().get_local().reshape((-1, self.dim_strain))    
        
        ft.setter(self.stress, np.array([self.get_stress(strain_table[i,:]) for i in range(self.n_gauss_points)]))
        ft.setter(self.tangent, np.array([self.get_tangent(strain_table[i,:]) for i in range(self.n_gauss_points)]))

        
    def get_stress(self, E):
        
        C = ft.GL2CG_np(self.conv.mandel2tensor_np(E))
        I1, I2, I3, J = ft.get_invariants_iso_np(C)
        Cbar = self.conv.tensor2mandel_np(J**(-2/3)*C)
        
        dpsidi1, dpsidi2 = self.get_dpsi(C)
        
        a1 = 2*(dpsidi1 + I1*dpsidi2)
        a2 = -2*dpsidi2
    
        return a1*self.conv.Id_mandel_np + a2*Cbar 

    
    def get_tangent(self, E):
        
        C = ft.GL2CG_np(self.conv.mandel2tensor_np(E))
        I1, I2, I3, J = ft.get_invariants_iso_np(C)
        Cbar = self.conv.tensor2mandel_np(J**(-2/3)*C)
        
        dpsidi1, dpsidi2 = self.get_dpsi(C) 
        d2psidi1i1, d2psidi2i2, d2psidi1di2 = self.get_d2psi(C) 

        # Taken from Holzapfels book, p. 262
        d1 = 4*(d2psidi1i1 + 2*I1*d2psidi1di2 + dpsidi2 + I1**2*d2psidi2i2)
        d2 = -4*(d2psidi1di2 + I1*d2psidi2i2)
        d3 = 4*d2psidi2i2
        d4 = -4*dpsidi2
        
        Id = self.conv.Id_mandel_np
        Id4_mandel = np.eye(self.dim_strain)
        sym = lambda A: 0.5*(A + A.T)
    
        D = (d1*np.outer(Id, Id) + 2*d2*sym(np.outer(Id, Cbar)) +
              d3*np.outer(Cbar, Cbar) + d4*Id4_mandel) 
        
        return D
    