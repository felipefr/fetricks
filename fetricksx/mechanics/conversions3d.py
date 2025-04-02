#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 19:30:27 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:03:12 2022

@author: felipefr


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""
import ufl
import numpy as np
from fetricksx.fenics.math_utils import symgrad

# unsymmetric notation (for 3d)
Id_unsym_df = ufl.as_vector(3*[1.0] + 6*[0.0])
Id_unsym_np = np.array(3*[1.0] + 6*[0.0])

def unsym2tensor_list(X):
    return [[X[0], X[3], X[5]],
            [X[4], X[1], X[7]],
            [X[6], X[8], X[2]]]

def tensor2unsym_list(X):
    return [X[0,0], X[1,1], X[2,2],  # Diagonal elements
            X[0,1], X[1,0],          # Off-diagonal (upper and lower for (0,1))
            X[0,2], X[2,0],          # Off-diagonal (upper and lower for (0,2))
            X[1,2], X[2,1]]          # Off-diagonal (upper and lower for (1,2))

def unsym2tensor_np(X):
    return np.array(unsym2tensor_list(X))

def tensor2unsym_np(X):
    return np.array(tensor2unsym_list(X))

def unsym2tensor(X):
    return ufl.as_tensor(unsym2tensor_list(X))

def tensor2unsym(X):
     return ufl.as_vector(tensor2unsym_list(X))

def tensor4th2unsym_list(X):
    return [[X[0,0,0,0], X[0,0,1,1], X[0,0,2,2], X[0,0,0,1], X[0,0,1,0], X[0,0,0,2], X[0,0,2,0], X[0,0,1,2], X[0,0,2,1]],
            [X[1,1,0,0], X[1,1,1,1], X[1,1,2,2], X[1,1,0,1], X[1,1,1,0], X[1,1,0,2], X[1,1,2,0], X[1,1,1,2], X[1,1,2,1]],
            [X[2,2,0,0], X[2,2,1,1], X[2,2,2,2], X[2,2,0,1], X[2,2,1,0], X[2,2,0,2], X[2,2,2,0], X[2,2,1,2], X[2,2,2,1]],
            [X[0,1,0,0], X[0,1,1,1], X[0,1,2,2], X[0,1,0,1], X[0,1,1,0], X[0,1,0,2], X[0,1,2,0], X[0,1,1,2], X[0,1,2,1]],
            [X[1,0,0,0], X[1,0,1,1], X[1,0,2,2], X[1,0,0,1], X[1,0,1,0], X[1,0,0,2], X[1,0,2,0], X[1,0,1,2], X[1,0,2,1]],
            [X[0,2,0,0], X[0,2,1,1], X[0,2,2,2], X[0,2,0,1], X[0,2,1,0], X[0,2,0,2], X[0,2,2,0], X[0,2,1,2], X[0,2,2,1]],
            [X[2,0,0,0], X[2,0,1,1], X[2,0,2,2], X[2,0,0,1], X[2,0,1,0], X[2,0,0,2], X[2,0,2,0], X[2,0,1,2], X[2,0,2,1]],
            [X[1,2,0,0], X[1,2,1,1], X[1,2,2,2], X[1,2,0,1], X[1,2,1,0], X[1,2,0,2], X[1,2,2,0], X[1,2,1,2], X[1,2,2,1]],
            [X[2,1,0,0], X[2,1,1,1], X[2,1,2,2], X[2,1,0,1], X[2,1,1,0], X[2,1,0,2], X[2,1,2,0], X[2,1,1,2], X[2,1,2,1]]]

def tensor4th2unsym(X):
    return ufl.as_tensor(tensor4th2unsym_list(X))

def tensor4th2unsym_np(X):
    return np.array(tensor4th2unsym_list(X))

def tr_unsym(X):
    return X[0] + X[1] + X[1]

def grad_unsym(v): # it was shown somehow to have better performance than doing it explicity
    return ufl.as_vector([v[0].dx(0), v[1].dx(1), v[2].dx(2), 
                         v[0].dx(1), v[1].dx(0),
                         v[0].dx(2), v[2].dx(0),
                         v[1].dx(2), v[2].dx(1)])
    
def macro_strain_unsym(i): 
    Eps_unsym = np.zeros((9,))
    Eps_unsym[i] = 1
    return unsym2tensor_np(Eps_unsym)

# mandel notation
sqrt2 = np.sqrt(2)
halfsqrt2 = 0.5*np.sqrt(2)

Id_mandel_df = ufl.as_vector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
Id_mandel_np = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

def mandel2tensor_np(X):
    return np.array([[X[0], halfsqrt2*X[5], halfsqrt2*X[4]],
                     [halfsqrt2*X[5], X[1], halfsqrt2*X[3]],
                     [halfsqrt2*X[4], halfsqrt2*X[3], X[2]] ])

def tensor2mandel_np(X):
    return np.array([X[0,0], X[1,1], X[2,2], 
                     halfsqrt2*(X[1,2] + X[2,1]), 
                     halfsqrt2*(X[0,2] + X[2,0]),
                     halfsqrt2*(X[0,1] + X[1,0])])

def mandel2tensor(X):
    return ufl.as_tensor( [[X[0], halfsqrt2*X[5], halfsqrt2*X[4]],
                          [halfsqrt2*X[5], X[1], halfsqrt2*X[3]],
                          [halfsqrt2*X[4], halfsqrt2*X[3], X[2]]])

def tensor2mandel(X):
    return ufl.as_vector([X[0,0], X[1,1], X[2,2], 
                         halfsqrt2*(X[1,2] + X[2,1]), 
                         halfsqrt2*(X[0,2] + X[2,0]),
                         halfsqrt2*(X[0,1] + X[1,0])])


def tensor4th2mandel(X):
    return ufl.as_tensor([ [X[0,0,0,0], X[0,0,1,1], X[0,0,2,2], sqrt2*X[0,0,1,2], sqrt2*X[0,0,2,0], sqrt2*X[0,0,0,1]],
                          [X[1,1,0,0], X[1,1,1,1], X[1,1,2,2], sqrt2*X[1,1,1,2], sqrt2*X[1,1,2,0], sqrt2*X[1,1,0,1]],
                          [X[2,2,0,0], X[2,2,1,1], X[2,2,2,2], sqrt2*X[2,2,1,2], sqrt2*X[2,2,2,0], sqrt2*X[2,2,0,1]],
                          [sqrt2*X[1,2,0,0], sqrt2*X[1,2,1,1], sqrt2*X[1,2,2,2], 2*X[1,2,1,2], 2*X[1,2,2,0], 2*X[1,2,0,1]],
                          [sqrt2*X[2,0,0,0], sqrt2*X[2,0,1,1], sqrt2*X[2,0,2,2], 2*X[2,0,1,2], 2*X[2,0,2,0], 2*X[2,0,0,1]],
                          [sqrt2*X[0,1,0,0], sqrt2*X[0,1,1,1], sqrt2*X[0,1,2,2], 2*X[0,1,1,2], 2*X[0,1,2,0], 2*X[0,1,0,1]] ])



def tensor4th2mandel_np(X):
    return np.array([ [X[0,0,0,0], X[0,0,1,1], X[0,0,2,2], sqrt2*X[0,0,1,2], sqrt2*X[0,0,2,0], sqrt2*X[0,0,0,1]],
                        [X[1,1,0,0], X[1,1,1,1], X[1,1,2,2], sqrt2*X[1,1,1,2], sqrt2*X[1,1,2,0], sqrt2*X[1,1,0,1]],
                        [X[2,2,0,0], X[2,2,1,1], X[2,2,2,2], sqrt2*X[2,2,1,2], sqrt2*X[2,2,2,0], sqrt2*X[2,2,0,1]],
                        [sqrt2*X[1,2,0,0], sqrt2*X[1,2,1,1], sqrt2*X[1,2,2,2], 2*X[1,2,1,2], 2*X[1,2,2,0], 2*X[1,2,0,1]],
                        [sqrt2*X[2,0,0,0], sqrt2*X[2,0,1,1], sqrt2*X[2,0,2,2], 2*X[2,0,1,2], 2*X[2,0,2,0], 2*X[2,0,0,1]],
                        [sqrt2*X[0,1,0,0], sqrt2*X[0,1,1,1], sqrt2*X[0,1,2,2], 2*X[0,1,1,2], 2*X[0,1,2,0], 2*X[0,1,0,1]] ])
                      

def tr_mandel(X):
    return X[0] + X[1] + X[2]


def symgrad_mandel(v):
    return tensor2mandel(symgrad(v))
    
# this is in mandel
def macro_strain_mandel(i): 
    Eps_Mandel = np.zeros((6,))
    Eps_Mandel[i] = 1
    return mandel2tensor_np(Eps_Mandel)


