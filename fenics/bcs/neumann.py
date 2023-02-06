#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:11:34 2023

@author: ffiguere
"""
import dolfin as df
import numpy as np

# buggy : sometimes crashes as eval is not available
# imposes dot(sig,n) = g  
class NeumannTensorSource(df.UserExpression):
    def __init__(self, mesh, g, **kwargs):
        self.mesh = mesh
        self.g = g
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, ufc_cell):
        cell = df.Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        values[:] = np.outer(self.g,n.array()[:self.mesh.geometric_dimension()]).flatten()
        
    def value_shape(self):
        return (self.mesh.geometric_dimension(),self.mesh.geometric_dimension())

# imposes dot(flux,n) = g  
class NeumannVectorSource(df.UserExpression):
    def __init__(self, mesh, g, **kwargs):
        self.mesh = mesh
        self.g = g
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = df.Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        values[:] = self.g*n.array()[:self.mesh.geometric_dimension()]
        
    def value_shape(self):
        return (self.mesh.geometric_dimension(),)
    


def NeumannVectorBC(W, t, mesh, flag):
    return [df.DirichletBC(W, NeumannTensorSource(mesh, t.values()) , mesh.boundaries, flag)]


def NeumannBC(W, t, mesh, flag):
    return [df.DirichletBC(W, NeumannVectorSource(mesh, t.values()) , mesh.boundaries, flag)]

# This is Neumann but when normal are aligned with the cartesian axes
def NeumannVectorBC_given_normal(W, t, normal, mesh, flag):
    sig_ = np.outer(t.values(), normal.values())
        
    return [df.DirichletBC(W, df.Constant(sig_), mesh.boundaries, flag)]