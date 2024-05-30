import os, sys
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import partial 

import numpy as np
from dolfinx import fem, io, mesh,plot
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI


def poisson(domain):
    #"Standard H^1(mesh) formulation of Poisson's equation." 
    
    Q = fem.functionspace(domain, ("CG", 1))
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    
    dx =  ufl.Measure('dx', domain = domain)
    a = ufl.inner(ufl.grad(p), ufl.grad(q))*dx
    f = fem.Constant(domain,1.0)
    L = f*q*dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(Q, fdim, boundary_facets)

    bcs = [fem.dirichletbc(fem.Constant(domain,0.0), boundary_dofs, Q)]

    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    p = problem.solve()

    return p
        
if __name__ == "__main__":
    
    n = 10
    
    domain = mesh.create_rectangle(MPI.COMM_WORLD,[[0.0, 0.0],[1,1]],[n,n], mesh.CellType.triangle, diagonal = mesh.DiagonalType.left)
    
    p = poisson(domain)

    print(np.linalg.norm(p.x.array))
