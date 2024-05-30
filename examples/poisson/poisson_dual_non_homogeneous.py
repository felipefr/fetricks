#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:57:41 2024

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

def primal(domain, param):
    Q = fem.functionspace(domain, ("CG", 2))
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    a = ufl.inner(ufl.grad(p), ufl.grad(q))*domain.dx
    f = fem.Constant(domain,param['f'])
    un_top = fem.Constant(domain,param['un_top'])
    un_bottom = fem.Constant(domain,param['un_bottom'])
    L = f*q*domain.dx + un_top*q*domain.ds(param['top']) + un_bottom*q*domain.ds(param['bottom'])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    left_dofs = fem.locate_dofs_topological(Q, fdim, domain.facets.find(param['left']))
    right_dofs = fem.locate_dofs_topological(Q, fdim, domain.facets.find(param['right']))

    bcs_D = [fem.dirichletbc(fem.Constant(domain,param['p_left']), left_dofs, Q),
             fem.dirichletbc(fem.Constant(domain,param['p_right']), right_dofs, Q)]

    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    problem = LinearProblem(a, L, bcs=bcs_D, petsc_options= petsc_options)
    p = problem.solve()

    return p

def dual(domain, param):
    #"Mixed H(div) x L^2 formulation of Poisson"
    
    Ve = basix.ufl.element("BDM", domain.domain.basix_cell(), 2, shape=(2,))
    Qe = basix.ufl.element("DG", domain.domain.basix_cell(), 1)
    W = fem.functionspace(domain.domain, basix.ufl.mixed_element([Ve, Qe]))
    
    normal = ufl.FacetNormal(domain)
    
    w = ufl.TrialFunction(W)
    dw = ufl.TestFunction(W)
    
    (u,p) = ufl.split(w)
    (v,q) = ufl.split(dw)

    a = (ufl.dot(u, v) + ufl.div(u)*q + ufl.div(v)*p)*domain.dx
    f = fem.Constant(domain,param['f'])
    p_left = fem.Constant(domain, param['p_left'])
    p_right = fem.Constant(domain, param['p_right'])
    L = -f*q*domain.dx + p_left*ufl.dot(v,normal)*domain.ds(param['left']) + p_right*ufl.dot(v,normal)*domain.ds(param['right'])
    
    W0 = W.sub(0)
    V, _ = W0.collapse()
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    bottom_dofs = fem.locate_dofs_topological((W0, V), fdim, domain.facets.find(param['bottom']))
    top_dofs = fem.locate_dofs_topological((W0, V), fdim, domain.facets.find(param['top']))
    
    
    x = ufl.SpatialCoordinate(domain)
    
    def g_top(x):
        values = np.zeros((2, x.shape[1]))
        values[1, :] = param['un_top']
        return values
    
    def g_bottom(x):
        values = np.zeros((2, x.shape[1]))
        print(x.shape[1])
        values[1, :] = -param['un_bottom']
        return values
    

    
    g_top_V = fem.Function(V)
    g_bottom_V = fem.Function(V)
    g_top_V.interpolate(g_top)
    g_bottom_V.interpolate(g_bottom)
    
    bcs_N = [ fem.dirichletbc(g_top_V, top_dofs, W0),
              fem.dirichletbc(g_bottom_V, bottom_dofs, W0)]
    
    
    # Note that without pc_factor_mat_solver_type it doesn't work. With mumps does not solve for big matrices
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    problem = LinearProblem  (a, L, bcs = bcs_N, petsc_options=petsc_options)
    
    w = problem.solve()

    return w
        
if __name__ == "__main__":
    
    param = {
    'f': 1.0,
    'un_top': 1.0,
    'un_bottom': 2.0,
    'p_left': 1.0,
    'p_right': 0.0,
    'bottom': 1,
    'left': 4,
    'right': 2,
    'top': 3,
    'mesh_file': 'unit_square.msh'
    }
    
    n = 100
    
    # domain, markers, facets = get_unit_square_mesh(n)
        
    domain = ft.Mesh(param['mesh_file'])
    
    
    start = timer()
    p = primal(domain, param)
    end = timer()
    print(end-start)
    start = timer()
    w = dual(domain, param)
    end = timer()
    print(end-start)
    
    p_dual = w.sub(1).collapse()

    print(np.linalg.norm(p.x.array))
    print(np.linalg.norm(p_dual.x.array))
    
    
    # comment if you don't want to use fetricks
    print(ft.error_L2(p, p_dual, 2, 4))
    
    # comment here if you have fetricks
    dx =  ufl.Measure('dx', domain = domain,  metadata={"quadrature_degree": 4})
    def L2Norm(u, comm):
        val = fem.assemble_scalar(fem.form(ufl.inner(u, u) * dx))
        return np.sqrt(comm.allreduce(val, op=MPI.SUM))
    
    print(L2Norm(p_dual-p, MPI.COMM_SELF))