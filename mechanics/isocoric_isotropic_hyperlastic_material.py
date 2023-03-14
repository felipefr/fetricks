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
        self.conv = {2: ft.conv2d, 3: ft.conv3d}[self.gdim]
            
    def create_internal_variables(self):
        self.stress = df.Function(self.W)
        self.strain = df.Function(self.W)
        self.tangent = df.Function(self.Wtan)
        
        projector_strain = ft.LocalProjector(self.W, self.dxm, self.strain)
        self.projector_list = {'strain' : projector_strain}

    def getE(self, u):
        return self.conv.symgrad_mandel(u) + 0.5*self.conv.tensor2mandel(df.grad(u).T*df.grad(u))

    def stress_op(self, E):
        return self.stress
        
    def tangent_op(self, de):
        return df.dot(self.tangent, de)
    
    def update(self, u):
        alpha_new = {'strain' : self.getE(u)}
        self.project_var(alpha_new)
        
        strain_table = self.strain.vector().get_local().reshape((-1, self.dim_strain))    
        
        ft.setter(self.stress, np.array([self.get_stress(strain_table[i,:]) for i in range(self.n_gauss_points)]))
        ft.setter(self.tangent, np.array([self.get_tangent(strain_table[i,:]) for i in range(self.n_gauss_points)]))


    def E2CG(self, E):
        return 2*self.conv.mandel2tensor_np(E) + np.eye(self.gdim)
    
    def get_invariants(self, C_):
        
        if(C_.shape[0] == 2):
            C = np.array([[C_[0,0], C_[0,1], 0], [C_[1,0], C_[1,1], 0], [0, 0, 1]])
        else:
            C = C_
        
        I3 = np.linalg.det(C)
        J = np.sqrt(I3)
        I1 = np.trace(C)
        I2 = 0.5*(np.trace(C)**2 - np.trace(C@C))
        
        return I1, I2, I3, J
    
    def get_invariants_iso(self, C_):
        I1, I2, I3, J = self.get_invariants(C_)
        
        return J**(-2/3)*I1, J**(-4/3)*I2, I3, J

        
    def get_stress(self, E):
        
        C = self.E2CG(E)
        I1, I2, I3, J = self.get_invariants_iso(C)
        Cbar = self.conv.tensor2mandel_np(J**(-2/3)*C)
        
        dpsidi1, dpsidi2 = self.get_dpsi(E) 
        
        a1 = 2*(dpsidi1 + I1*dpsidi2)
        a2 = -2*dpsidi2
    
        return a1*self.conv.Id_mandel_np + a2*Cbar 

    
    def get_tangent(self, E):
        
        C = self.E2CG(E)
        I1, I2, I3, J = self.get_invariants_iso(C)
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
    