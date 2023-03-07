#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:18:46 2022

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
from .material_model import materialModel , materialModelExpression

import fetricks as ft

# mandel2tensor
# LocalProjector

class hyperelasticModel(materialModel):
        
    def param_parser(self, param):
        
        if('lamb' in param.keys()):
            self.lamb = param['lamb']
            self.mu = param['mu']
            
        else: 
            E = param['E']
            nu = param['nu']
            self.lamb = E*nu/(1+nu)/(1-2*nu)
            self.mu = E/2./(1+nu)
            
        self.alpha = param['alpha']  if 'alpha' in param.keys() else 0.0
        

    def create_internal_variables(self):
        self.sig = df.Function(self.W)
        self.eps = df.Function(self.W)
        
        projector_eps = ft.LocalProjector(self.W, self.dxm, self.eps)
        projector_sig = ft.LocalProjector(self.W, self.dxm, self.sig)
        
        self.projector_list = {'eps' : projector_eps,  'sig' : projector_sig}

    def stress(self, lamb_, mu_, eps): # elastic (I dont know why for the moment) # in mandel format
        return lamb_*ft.tr_mandel(eps)*ft.Id_mandel_df + 2*mu_*eps
    
    def epseps_de(self, de):
        return df.inner(self.eps, de)*self.eps
    
    def tangent(self, de): # de have to be in mandel notation
        ee = df.inner(self.eps, self.eps)
        tre2 = ft.tr_mandel(self.eps)**2.0
        
        lamb_ = self.lamb*( 1 + 3*self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        return self.stress(lamb_, mu_, de)  + 4*self.mu*self.alpha*self.epseps_de(de)

    def update(self, epsnew):
        
        ee = df.inner(epsnew, epsnew)
        tre2 = ft.tr_mandel(epsnew)**2.0
        
        lamb_ = self.lamb*( 1 + self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        alpha_new = {'eps' : epsnew, 'sig': self.stress(lamb_, mu_, epsnew)}
        self.project_var(alpha_new)
        
        
# Constant materials params
class hyperelasticModelExpression(materialModelExpression):
    
    def __init__(self, mesh, param, deg_stress = 0, dim_strain = 3):
        super().__init__(mesh, param, deg_stress, dim_strain)
    
 
    def param_parser(self, param):
        
        if('lamb' in param.keys()):
            self.lamb = param['lamb']
            self.mu = param['mu']
            
        else: 
            E = param['E']
            nu = param['nu']
            self.lamb = E*nu/(1+nu)/(1-2*nu)
            self.mu = E/2./(1+nu)
            
        self.alpha = param['alpha']  if 'alpha' in param.keys() else 0.0
        
    
    def pointwise_stress(self, e, cell = None): # in mandel format
    
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb*( 1 + self.alpha*tre2)
        mu_star = self.mu*( 1 + self.alpha*ee ) 
        
        return lamb_star*(e[0] + e[1])*ft.Id_mandel_np + 2*mu_star*e
    
    
    def pointwise_tangent(self, e, cell = None): # in mandel format
        
        ee = np.dot(e,e)
        tre2 = (e[0] + e[1])**2.0
        
        lamb_star = self.lamb*( 1 + 3*self.alpha*tre2)
        mu_star = self.mu*( 1 + self.alpha*ee ) 
        
        D = 4*self.mu*self.alpha*np.outer(e,e)
    
        D[0,0] += lamb_star + 2*mu_star
        D[1,1] += lamb_star + 2*mu_star
        D[0,1] += lamb_star
        D[1,0] += lamb_star
        D[2,2] += 2*mu_star

        return D
