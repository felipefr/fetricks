#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:46:04 2024

@author: felipe
"""

import os, sys
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import partial 

import numpy as np
from dolfinx import fem, io, mesh,plot
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI
import gmsh

# n : number of divisions per edge
def get_unit_square_mesh(n): 
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
    geom = gmsh.model.geo
    
    p = []
    p.append(geom.add_point(0.0, 0.0, 0.0))
    p.append(geom.add_point(1.0, 0.0, 0.0))
    p.append(geom.add_point(1.0, 1.0, 0.0))
    p.append(geom.add_point(0.0, 1.0, 0.0))

        
    l = []
    l.append(geom.add_line(p[0], p[1]))
    l.append(geom.add_line(p[1], p[2]))
    l.append(geom.add_line(p[2], p[3]))
    l.append(geom.add_line(p[3], p[0]))
    
    ll = [geom.add_curve_loop([l[0], l[1], l[2], l[3]])]
    s = [geom.add_plane_surface(ll)]
    
    geom.synchronize()
    
    for li in l:
        gmsh.model.mesh.set_transfinite_curve(li, n+1)
    
    gmsh.model.mesh.set_transfinite_surface(s[0],arrangement="Left")
    
    gmsh.model.add_physical_group(2, s, 0)
    for i in range(4):
        gmsh.model.add_physical_group(1, [l[i]], i+1) # topology, list objects, flag
        
    gmsh.model.mesh.generate(dim=2)
    gmsh.write("unit_square.msh")
    
    domain, markers, facets = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    
    gmsh.finalize()
    
    return domain, markers, facets


def poisson(domain, markers, facets):
    Q = fem.functionspace(domain, ("CG", 1))
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

    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    p = problem.solve()

    return p
        
if __name__ == "__main__":
    
    n = 10
    
    domain, markers, facets = get_unit_square_mesh(n)
    
    p = poisson(domain, markers, facets)

    print(np.linalg.norm(p.x.array))
