#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:24:57 2024

@author: felipe
"""

import os, sys
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import partial 

import numpy as np
from dolfinx import fem, io, mesh,plot
import basix
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI
import gmsh

# comment or change here if you don't want to use fetricks or it is somewhere located
sys.path.append("/home/felipe/sources/fetricksx")
import fetricksx as ft

from poisson_gmsh import get_unit_square_mesh


def primal(domain, markers, facets):
    Q = fem.functionspace(domain, ("CG", 2))
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    
    dx =  ufl.Measure('dx', domain = domain)
    a = ufl.inner(ufl.grad(p), ufl.grad(q))*dx
    f = fem.Constant(domain,1.0)
    L = f*q*dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_dofs = np.concatenate(tuple([fem.locate_dofs_topological(Q, fdim, facets.find(i+1)) for i in range(4)]))

    bcs = [fem.dirichletbc(fem.Constant(domain,0.0), boundary_dofs, Q)]

    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    problem = LinearProblem(a, L, bcs=bcs, petsc_options= petsc_options)
    p = problem.solve()

    return p

def dual(domain, markers, facets):
    #"Mixed H(div) x L^2 formulation of Poisson"
    
    Ve = basix.ufl.element("RT", domain.basix_cell(), 2, shape=(2,))
    Qe = basix.ufl.element("DG", domain.basix_cell(), 1)
    W = fem.functionspace(domain, basix.ufl.mixed_element([Ve, Qe]))
    

    w = ufl.TrialFunction(W)
    dw = ufl.TestFunction(W)
    
    (u,p) = ufl.split(w)
    (v,q) = ufl.split(dw)
    
    dx =  ufl.Measure('dx', domain = domain)
    a = (ufl.dot(u, v) + ufl.div(u)*q + ufl.div(v)*p)*dx
    f = fem.Constant(domain, -1.0)
    L = f*q*dx
    
    # Note that without pc_factor_mat_solver_type it doesn't work. With mumps does not solve for big matrices^
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "umfpack"}
    problem = LinearProblem(a, L, petsc_options=petsc_options)
    
    w = problem.solve()

    return w
        
if __name__ == "__main__":
    
    n = 100
    
    domain, markers, facets = get_unit_square_mesh(n)
    
    p = primal(domain, markers, facets)
    w = dual(domain, markers, facets)
    
    p_dual = w.sub(1).collapse()

    print(np.linalg.norm(p.x.array))
    print(np.linalg.norm(p_dual.x.array))
    
    
    # comment if you don't want to use fetricks
    print(ft.error_L2(p, p_dual, 2, 4))
    
    # comment here if you have fetricks
    # dx =  ufl.Measure('dx', domain = domain,  metadata={"quadrature_degree": 4})
    # def L2Norm(u, comm):
    #     val = fem.assemble_scalar(fem.form(ufl.inner(u, u) * dx))
    #     return np.sqrt(comm.allreduce(val, op=MPI.SUM))
    
    # print(L2Norm(p_dual-p, MPI.COMM_SELF))