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
    
    Q = df.FunctionSpace(mesh, "CG", 1)
    p = df.TrialFunction(Q)
    q = df.TestFunction(Q)

    a = ufl.inner(ufl.grad(p), ufl.grad(q))*df.dx
    f = df.Constant(1.0)
    L = f*q*df.dx

    bc = df.DirichletBC(Q, 0.0, "on_boundary")

    A, b = df.assemble_system(a, L, bc)
    
    p = df.Function(Q)
    solver = df.LUSolver(A)
    solver.solve(p.vector(), b)

    return p
        
if __name__ == "__main__":
    
    n = 10
    mesh = df.UnitSquareMesh(n, n)
    
    p = primal(mesh)

    print(np.linalg.norm(p.vector().get_local()))
