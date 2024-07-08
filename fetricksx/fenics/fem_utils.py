#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 12:13:13 2024

@author: felipefr
"""

from basix.ufl import element, mixed_element
from dolfinx import fem
import ufl
import basix
import numpy as np

# fe_params : tuple family, degree
def mixed_functionspace(msh, fe_params): 
    FEs = [element(fe[0], msh.domain.basix_cell(), fe[1]) for fe in fe_params]
    Wh = fem.functionspace(msh, mixed_element(FEs))
    return Wh


class CustomQuadratureSpace:
    
    def __init__(self, mesh, dim, degree_quad = None):
        self.degree_quad = degree_quad
        self.mesh = mesh
        self.basix_cell = self.mesh.basix_cell()
        
        self.dxm = ufl.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": self.degree_quad, "quadrature_scheme": "default"})
        self.W0e = basix.ufl.quadrature_element(self.basix_cell, degree=self.degree_quad, scheme = "default", value_shape= ())
        self.We = basix.ufl.quadrature_element(self.basix_cell, degree=self.degree_quad, scheme = "default", value_shape = (dim,))
        self.space = fem.functionspace(self.mesh, self.We)       
        self.scalar_space = fem.functionspace(self.mesh, self.W0e)
        basix_celltype = getattr(basix.CellType, self.mesh.topology.cell_type.name)
        points, weights = basix.make_quadrature(basix_celltype, self.degree_quad)
        self.eval_points = points
        self.weights = weights
        self.nq_cell = len(self.eval_points) # number of quadrature points per cell 
        self.nq_mesh = self.mesh.num_cells*self.nq_cell # number of quadrature points 
        
        
class QuadratureEvaluator:
    
    def __init__(self, ufl_expr, storage_array, mesh, W):
        self.mesh = mesh
        self.eval_points = W.eval_points
        self.cells = self.get_indexes_cells()
        self.femexp = fem.Expression(ufl_expr, self.eval_points)
        self.storage_array = storage_array
        
    def get_indexes_cells(self):
        map_c = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        return np.arange(0, num_cells, dtype=np.int32)
    
    def __call__(self):
        return self.femexp.eval(self.mesh, self.cells, self.storage_array)