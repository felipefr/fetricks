#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:58:05 2024

@author: felipe
"""

import sys
import dolfin as df
import ufl
import numpy
from timeit import default_timer as timer
import numpy as np

sys.path.append("/home/felipe/sources/fetricks")
import fetricks as ft

def primal(mesh, param):
    #"Standard H^1(mesh) formulation of Poisson's equation." 
    
    Q = df.FunctionSpace(mesh, "CG", 2)
    p = df.TrialFunction(Q)
    q = df.TestFunction(Q)

    a = ufl.inner(ufl.grad(p), ufl.grad(q))*mesh.dx
    f = df.Constant(param['f']) 
    un_top = df.Constant(param['un_top'])
    un_bottom = df.Constant(param['un_bottom'])
    L = f*q*mesh.dx + un_top*q*mesh.ds(param['top']) + un_bottom*q*mesh.ds(param['bottom'])

    bcs_D = [ 
        df.DirichletBC(Q, param['p_left'], mesh.boundaries, param['left']),
        df.DirichletBC(Q, param['p_right'], mesh.boundaries, param['right'])
        ]
        

    A, b = df.assemble_system(a, L, bcs_D)
    
    p = df.Function(Q)
    df.solve(A, p.vector(), b)

    return p

def dual(mesh, param):
    #"Mixed H(div) x L^2 formulation of Poisson"
    
    V = df.FiniteElement("RT", mesh.ufl_cell(), 2)
    Q = df.FiniteElement("DG", mesh.ufl_cell(), 1)
    W = df.FunctionSpace(mesh, V*Q)
    normal = df.FacetNormal(mesh)
    
    (u, p) = df.TrialFunctions(W)
    (v, q) = df.TestFunctions(W)

    a = (df.dot(u, v) + df.div(u)*q + df.div(v)*p)*mesh.dx
    f = df.Constant(param['f']) 
    p_left = df.Constant(param['p_left'])
    p_right = df.Constant(param['p_right'])
    L = -f*q*mesh.dx + p_left*df.dot(v,normal)*mesh.ds(param['left']) + p_right*df.dot(v,normal)*mesh.ds(param['right'])
    
    
    bcs_N = [ ft.NeumannScalarBC(W.sub(0), df.Constant(param['un_bottom']), mesh, param['bottom'], op="cpp"),
              ft.NeumannScalarBC(W.sub(0), df.Constant(param['un_top']), mesh, param['top'], op="cpp")]

    
    A, b = df.assemble_system(a, L, bcs_N)
    w = df.Function(W)
    df.solve(A, w.vector(), b)

    return w

        
if __name__ == "__main__":
    
    param = {
    'f': 1,
    'un_top': 0.0,
    'un_bottom': 1.0,
    'p_left': 1.0,
    'p_right': 0.0,
    'bottom': 2,
    'left': 0,
    'right': 1,
    'top': 3
    }
    
    n = 100
    mesh = df.UnitSquareMesh(n, n)
    mesh = ft.Mesh(mesh)
    
    p = primal(mesh, param)
    w = dual(mesh, param)
    
    (u_dual, p_dual) = w.split(deepcopy=True)
    
    print(np.linalg.norm(p.vector().get_local()))
    print(np.linalg.norm(p_dual.vector().get_local()))
    dx =  ufl.Measure('dx', domain = mesh,  metadata={"quadrature_degree": 2})
    print(np.sqrt(df.assemble(df.inner(p-p_dual,p-p_dual)*dx)))
