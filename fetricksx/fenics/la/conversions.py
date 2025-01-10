#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 21 08:31:08 2023

@author: ffiguere 


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import ufl
import numpy as np
# import numba

# row by row convention
# as_sym_tensor = lambda a: ufl.as_tensor( [ [ a[0], a[1], a[2]] , [a[1] , a[3], a[4]] , [a[2] , a[4], a[5]] ])
# ind_sym_tensor = np.array([0, 1, 2, 4, 5, 8])

# collect_stress = lambda m, e: np.array( [ m[i].getStress(e[i,:]) for i in range(len(m))] ).flatten()
# collect_tangent = lambda m, e: np.array( [ m[i].getTangent(e[i,:]).flatten()[ind_sym_tensor] for i in range(len(m))] ).flatten()


# following functions using diagonal + non_diagonal convention (non mandel)
ind_sym_tensor_3x3 = np.array([0, 4, 8, 5, 2, 1]).astype('int')

def as_sym_tensor_3x3(a):
    return ufl.as_tensor( [[ a[0], a[5], a[4]] , [a[5] , a[1], a[3]] , [a[4] , a[3], a[2]]])

#@numba.jit(nopython=True)
def as_sym_tensor_3x3_np(a):
    return np.array( [[ a[0], a[5], a[4]] , [a[5] , a[1], a[3]] , [a[4] , a[3], a[2]]])

#@numba.jit(nopython=True)
def sym_flatten_3x3_np(A):
    return 0.5*(A + A.T).flatten()[ind_sym_tensor_3x3]

# --------------------------------- 

def flatgrad_2x2(v):
    return ufl.as_vector([v[0].dx(0), v[0].dx(1), v[1].dx(0), v[1].dx(1)])

def flatsymgrad_2x2(v):
    return as_flatten_2x2(ufl.sym(ufl.grad(v)))


# flatten([v x A]) = as_cross(A)*v ([v x A]_ij = e_ijk v_k A_lj ) (Bonnet's definition) 
def as_cross_3x3(A):
    return ufl.as_tensor([[0, A[2,0], -A[1,0]],
                        [0, A[2,1], -A[1,1]],
                        [0, A[2,2], -A[1,2]],
                        [-A[2,0], 0, A[0,0]],
                        [-A[2,1], 0, A[0,1]],
                        [-A[2,2], 0, A[0,2]],
                        [A[1,0], -A[0,0], 0],
                        [A[1,1], -A[0,1], 0],
                        [A[1,2], -A[0,2], 0]])
    

# flatten([v x A]) = as_cross(A)*v ([v x A]_ij = e_ijk v_k A_lj ) (Bonnet's definition)
def as_cross_2x2(A):
    return ufl.as_tensor([[A[1,0]],
                         [A[1,1]],
                         [-A[0,0]],
                         [-A[0,1]]])

# a*(A[0,1] - A[1,0])  = inner(as_skew(a) , A) 
def as_skew_2x2(a):
    return ufl.as_tensor([[0, a], [-a, 0]])
                                               
def as_flatten_3x3(A):
    return ufl.as_vector([A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2], A[2,0], A[2,1], A[2,2]])

def as_flatten_2x2(A):
    return ufl.as_vector([A[0,0], A[0,1], A[1,0], A[1,1]])


def as_unflatten_2x2(a):
    return ufl.as_tensor([[a[0], a[1]], [a[2], a[3]]])


# numpy version of as_cross fenics 
def as_cross_3x3_np(A):
     (x,y) = np.shape(A)
     Ax = np.zeros((x*y, y))
     Ax[3:6,0] = -A[2,0:3]
     Ax[6:9,0] = A[1,0:3]
     Ax[0:3,1] = A[2,0:3]
     Ax[0:3,2] = -A[1,0:3]
     Ax[3:6,2] = A[0,0:3]
     Ax[6:9,1] = -A[0,0:3]
     return Ax

def as_cross_2x2_np(A):
    return np.array([A[1,0], A[1,1], -A[0,0],-A[0,1]])
                        