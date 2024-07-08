#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:08:19 2024

@author: felipe
"""
import ufl
from dolfinx import fem
from mpi4py import MPI
import numpy as np

# NON-FLATTENED FUNCTIONS
def symgrad(v): 
    return ufl.sym(ufl.grad(v))

# Vectorial and Tensorial integrals (Fenics integrals are scalars by default)
def integral(u,dx, mesh, shape):
    
    n = len(shape)
    I = np.zeros(shape)
        
    if(n == 0):
        form = fem.form(u * dx)
        integral_local = fem.assemble_scalar(form)
        I = mesh.comm.allreduce(integral_local, op=MPI.SUM)
 
    elif(n == 1):
        for i in range(shape[0]):
            I[i] += integral(u[i], dx, mesh, shape = ())
            
    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                I[i,j] += integral(u[i,j], dx, mesh, shape = ())
    
    else:
        print('not implement for higher order integral')
        
    
    return I
