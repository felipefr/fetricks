#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:11:34 2023

@author: ffiguere

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""
import dolfin as df
import numpy as np


        # cell = df.Cell(self.mesh, ufc_cell.index)
        # n = cell.normal(ufc_cell.local_facet)
        # values[:] = np.outer(self.g,n.array()[:self.mesh.geometric_dimension()]).flatten()

code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/geometry/Point.h>

typedef Eigen::VectorXd npArray;
typedef Eigen::VectorXi npArrayInt;
typedef dolfin::Mesh dfMesh;

class BoundarySource : public dolfin::Expression {
  public:
     
    dfMesh mesh;
    npArray gs;
    int nc;
    
    BoundarySource(dfMesh m, npArray g, int nc) : dolfin::Expression(nc), mesh(mesh), gs(g), nc(nc) { }

    void eval(Eigen::Ref<Eigen::VectorXd> values,
                      Eigen::Ref<const Eigen::VectorXd> x,
                      const ufc::cell& cell) const {
        
        dolfin::Cell cell_local(mesh, cell.index);
        dolfin::Point normal = cell_local.normal(cell.local_facet);         
        int gdim = gs.size();    
    
        for(int i = 0; i<gdim ; i++){ 
            for(int j = 0; j<gdim ; j++) values[i*gdim + j] = normal.coordinates()[i]*gs[j]; 
        }
    }
                      
                    
};

PYBIND11_MODULE(SIGNATURE, m) {
    pybind11::class_<BoundarySource, std::shared_ptr<BoundarySource>, dolfin::Expression>
    (m, "BoundarySource")
    .def(pybind11::init< dfMesh, npArray, int>())
    .def("__call__", &BoundarySource::eval);
}
"""


compCode = df.compile_cpp_code(code)
BoundarySourceCpp = lambda gdim, mesh, g: df.CompiledExpression(compCode.BoundarySource(mesh, np.zeros(2), g, gdim*gdim))

# buggy : sometimes crashes as eval is not available
# imposes dot(sig,n) = g  
class NeumannTensorSource(df.UserExpression):
    def __init__(self, mesh, g, **kwargs):
        self.mesh = mesh
        self.g = g
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, ufc_cell):
        if(ufc_cell.local_facet>-1):
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
        if(ufc_cell.local_facet>-1):
            cell = df.Cell(self.mesh, ufc_cell.index)
            n = cell.normal(ufc_cell.local_facet)
            values[:] = self.g*n.array()[:self.mesh.geometric_dimension()]
            
    def value_shape(self):
        return (self.mesh.geometric_dimension(),)
    

class SelectBoundary(df.SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary 
      
def NeumannVectorBC(W, t, mesh, flag):
    g = NeumannTensorSource(mesh, t.values()) 
    # g = df.interpolate(g, df.VectorFunctionSpace(mesh, "BDM", 2, restriction=SelectBoundary()))
    # g = BoundarySourceCpp(2, mesh, t.values())
    return [df.DirichletBC(W, g , mesh.boundaries, flag)]


def NeumannBC(W, t, mesh, flag):
    g = NeumannVectorSource(mesh, t.values())
    return [df.DirichletBC(W, g, mesh.boundaries, flag)]

# This is Neumann but when normal are aligned with the cartesian axes
def NeumannVectorBC_given_normal(W, t, normal, mesh, flag):
    sig_ = np.outer(t.values(), normal.values())
        
    return [df.DirichletBC(W, df.Constant(sig_), mesh.boundaries, flag)]