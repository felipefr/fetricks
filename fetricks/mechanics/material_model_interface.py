#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:09:55 2022

@author: felipe


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

# TO DO: Compatibilise intervaces

import abc
import dolfin as df
import numpy as np
import fetricks as ft


class materialModel(metaclass=abc.ABCMeta):
    
    def __init__(self, mesh, param, deg_stress = 0, dim_strain = 3):
        
        self.deg_stress = deg_stress
        self.mesh = mesh
        self.dim_strain = dim_strain
        self.param_parser(param)
        self.W0, self.W, self.Wtan = ft.create_quadrature_spaces_mechanics(mesh, deg_stress, self.dim_strain)
        # self.W0, self.W, self.Wtan = ft.create_DG_spaces_mechanics(mesh, deg_stress, self.dim_strain)
        
        self.n_gauss_points = self.W.dim()//self.dim_strain
        
        metadata = {"quadrature_degree": deg_stress, 'quadrature_scheme': 'default' }
        self.dxm = df.dx(metadata=metadata)
    
        self.create_internal_variables()


    # optional
    def get_dpsi(self, E):
        pass
    
    def get_d2psi(self,E):
        pass
    
    @abc.abstractmethod 
    def stress_op(self, e):
        pass
    
    @abc.abstractmethod 
    def tangent_op(self, e):
        pass
    
    @abc.abstractmethod  
    def create_internal_variables(self):
        pass
    
    @abc.abstractmethod    
    def param_parser(self, param):
        pass
    
    @abc.abstractmethod 
    def update(self, e):
        pass
    
    def project_var(self, AA):
        for label in AA.keys(): 
            self.projector_list[label](AA[label])



class materialModelExpression(metaclass=abc.ABCMeta):
    def __init__(self, mesh, param, deg_stress = 0, dim_strain = 3):
        
        self.param_parser(param)
        self.W0, self.W, self.Wtan = ft.create_quadrature_spaces_mechanics(mesh, deg_stress, dim_strain)
        
        metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
        self.dxm = df.dx(metadata=metadata)
    
        self.strain = df.Function(self.W) 
        self.projector = ft.LocalProjector(self.W, self.dxm, self.strain)
        
        self.size_strain = self.W.num_sub_spaces()
        self.size_tan = self.Wtan.num_sub_spaces()
        self.size_tan_sqrt = int(np.sqrt(self.size_tan))
        
        self.stress = ft.genericGaussPointExpression(self.strain, self.pointwise_stress , (self.size_strain,))
        self.tangent = ft.genericGaussPointExpression(self.strain, self.pointwise_tangent , (6, ))


    @abc.abstractmethod     
    def param_parser(self, param):
        pass
    
    @abc.abstractmethod     
    def tangent_op(self, e):
        pass
    
    @abc.abstractmethod 
    def pointwise_stress(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        pass
    
    @abc.abstractmethod 
    def pointwise_tangent(self, e, cell = None): # elastic (I dont know why for the moment) # in mandel format
        pass
    
    def update(self, e):
        self.projector(e)
