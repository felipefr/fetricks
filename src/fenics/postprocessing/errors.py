#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:05:05 2024

@author: felipe
"""

import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI
import basix

def error_L2(u1, u2, degree_raise, quad_degree):
    # Create higher order function space
    degree = u1.function_space.ufl_element().degree
    family = u1.function_space.ufl_element().family_name
    domain = u1.function_space.mesh
    # discontinuity is useful for Hdiv elements
    We = basix.ufl.element(family, domain.basix_cell(), degree + degree_raise, discontinuous = True)
    W = fem.functionspace(domain, We)
    
    # Interpolate approximate solution
    u1_W = fem.Function(W)
    u1_W.interpolate(u1)

    u2_W = fem.Function(W)
    u2_W.interpolate(u2)

    # Compute the error in the higher order function space
    e_W = fem.Function(W)
    e_W.x.array[:] = u1_W.x.array - u2_W.x.array

    # Integrate the error
    dx =  ufl.Measure('dx', domain = domain,  metadata={"quadrature_degree": quad_degree})
    error = fem.form(ufl.inner(e_W, e_W) * dx)
    error_local = fem.assemble_scalar(error)
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
    
    return np.sqrt(error_global)
