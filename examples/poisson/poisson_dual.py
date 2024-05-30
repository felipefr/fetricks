#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:04:22 2024

@author: felipe
"""

import dolfin as df
import ufl
import numpy
from timeit import default_timer as timer
import numpy as np

def primal(mesh):
    #"Standard H^1(mesh) formulation of Poisson's equation." 
    
    Q = df.FunctionSpace(mesh, "CG", 2)
    p = df.TrialFunction(Q)
    q = df.TestFunction(Q)

    a = ufl.inner(ufl.grad(p), ufl.grad(q))*df.dx
    f = df.Constant(1.0)
    L = f*q*df.dx

    bc = df.DirichletBC(Q, 0.0, "on_boundary")

    A, b = df.assemble_system(a, L, bc)
    
    p = df.Function(Q)
    df.solve(A, p.vector(), b)

    return p

def dual(mesh):
    #"Mixed H(div) x L^2 formulation of Poisson"
    
    V = df.FiniteElement("RT", mesh.ufl_cell(), 2)
    Q = df.FiniteElement("DG", mesh.ufl_cell(), 1)
    W = df.FunctionSpace(mesh, V*Q)

    (u, p) = df.TrialFunctions(W)
    (v, q) = df.TestFunctions(W)

    a = (df.dot(u, v) + df.div(u)*q + df.div(v)*p)*df.dx
    f = df.Constant(-1.0)
    L = f*q*df.dx

    A = df.assemble(a)
    b = df.assemble(L)
    w = df.Function(W)
    
    df.solve(A, w.vector(), b)

    return w

        
if __name__ == "__main__":
    
    n = 100
    mesh = df.UnitSquareMesh(n, n)
    
    p = primal(mesh)
    w = dual(mesh)
    
    (u_dual, p_dual) = w.split(deepcopy=True)
    
    print(np.linalg.norm(p.vector().get_local()))
    print(np.linalg.norm(p_dual.vector().get_local()))
    dx =  ufl.Measure('dx', domain = mesh,  metadata={"quadrature_degree": 2})
    print(np.sqrt(df.assemble(df.inner(p-p_dual,p-p_dual)*dx)))
