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



codeVectorSource = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/geometry/Point.h>

typedef Eigen::VectorXd npArray;
typedef Eigen::VectorXi npArrayInt;
typedef dolfin::Mesh dfMesh;
typedef dolfin::Cell dfCell;

class NeumannVectorSourceCpp : public dolfin::Expression {
  public:
     
    dfMesh mesh;
    double gs;
    int gdim;
    
    NeumannVectorSourceCpp(dfMesh m, double g, int gdim) : dolfin::Expression(gdim){
        mesh = m;
        gs = g;
        gdim = gdim;
        }

    void eval(Eigen::Ref<Eigen::VectorXd> values,
              Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const {

        int i = cell.local_facet;
        if(i > -1){                  
            dfCell c(mesh, cell.index);
            values << gs*c.normal(i, 0), gs*c.normal(i, 1) ;
        }
    }                      
                    
};
                  
class NeumannTensorSourceCpp : public dolfin::Expression {
  public:
     
    dfMesh mesh;
    npArray gs;
    int gdim;
    
    NeumannTensorSourceCpp(dfMesh m, npArray g, int gdim) : dolfin::Expression(gdim,gdim){
        mesh = m;
        gs = g;
        gdim = gdim;
        }

    void eval(Eigen::Ref<Eigen::VectorXd> values,
                      Eigen::Ref<const Eigen::VectorXd> x,
                      const ufc::cell& cell) const {
        
        int i = cell.local_facet;
        if(i > -1){                  
            dfCell c(mesh, cell.index);
            values << gs[0]*c.normal(i, 0), gs[0]*c.normal(i, 1), gs[1]*c.normal(i, 0), gs[1]*c.normal(i, 1) ;
        }
    }
    
                      
                    
};

PYBIND11_MODULE(SIGNATURE, m) {
    pybind11::class_<NeumannVectorSourceCpp, std::shared_ptr<NeumannVectorSourceCpp>, dolfin::Expression>
    (m, "NeumannVectorSourceCpp")
    .def(pybind11::init< dfMesh, double, int>())
    .def("__call__", &NeumannVectorSourceCpp::eval);

    pybind11::class_<NeumannTensorSourceCpp, std::shared_ptr<NeumannTensorSourceCpp>, dolfin::Expression>
    (m, "NeumannTensorSourceCpp")
    .def(pybind11::init< dfMesh, npArray, int>())
    .def("__call__", &NeumannTensorSourceCpp::eval);
}
"""

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
    

@static_vars(compCode = None)
def NeumannTensorSourceCpp(mesh, g, degree = 1):
    if(not NeumannTensorSourceCpp.compCode):
        NeumannTensorSourceCpp.compCode = df.compile_cpp_code(codeVectorSource)
    return df.CompiledExpression(NeumannTensorSourceCpp.compCode.NeumannTensorSourceCpp(mesh, g, mesh.geometric_dimension()), degree = degree)

@static_vars(compCode = None)
def NeumannVectorSourceCpp(mesh, g, degree = 1):
    if(not NeumannVectorSourceCpp.compCode):
        NeumannVectorSourceCpp.compCode = df.compile_cpp_code(codeVectorSource)
    return df.CompiledExpression(NeumannVectorSourceCpp.compCode.NeumannVectorSourceCpp(mesh, g, mesh.geometric_dimension()), degree = degree)

    
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
      
def NeumannVectorBC(W, t, mesh, flag, op = 'cpp'):
    if(op == "cpp"):
        g = NeumannTensorSourceCpp(mesh, t.values())
    else:
        g = NeumannTensorSource(mesh, t.values())
    return [df.DirichletBC(W, g , mesh.boundaries, flag)]


def NeumannBC(W, t, mesh, flag, op = 'cpp'):
    g = NeumannVectorSource(mesh, t.values())
    if(op == "cpp"):
        g = NeumannVectorSourceCpp(mesh, t.values())
    else:
        g = NeumannVectorSource(mesh, t.values())

    return [df.DirichletBC(W, g, mesh.boundaries, flag)]

# This is Neumann but when normal are aligned with the cartesian axes
def NeumannVectorBC_given_normal(W, t, normal, mesh, flag):
    sig_ = np.outer(t.values(), normal.values())
        
    return [df.DirichletBC(W, df.Constant(sig_), mesh.boundaries, flag)]