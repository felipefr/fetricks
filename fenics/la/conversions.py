#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 21 08:31:08 2023

@author: ffiguere 

"""

import dolfin as df
import numpy as np


# flatten([v x A]) = as_cross(A)*v ([v x A]_ij = e_ijk v_k A_lj ) (Bonnet's definition) 
def as_cross_3x3(A):
    return df.as_tensor([[0, A[2,0], -A[1,0]],
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
    return df.as_tensor([[A[1,0]],
                         [A[1,1]],
                         [-A[0,0]],
                         [-A[0,1]]])
                        
def as_flatten_3x3(A):
    return df.as_vector([A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2], A[2,0], A[2,1], A[2,2]])

def as_flatten_2x2(A):
    return df.as_vector([A[0,0], A[0,1], A[1,0], A[1,1]])


def as_unflatten_2x2(a):
    return df.as_tensor([[a[0], a[1]], [a[2], a[3]]])


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
                        