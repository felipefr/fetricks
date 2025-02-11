#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:56:45 2023

@author: ffiguere
"""

import numpy as np
import dolfin as df
import fetricks as ft
import scipy as sp

# Green-Lagrange (E) to Cauchy-Green (C)
# E is mandel
def GL2CG_np(E):
    return 2*E + np.eye(E.shape[0])

def plane_strain_CG_np(C):
    return np.array([[C[0,0], C[0,1], 0], [C[1,0], C[1,1], 0], [0, 0, 1]])

# C is full
def get_invariants_np(C, constraint = 'PLANE_STRAIN'):
    
    if(C.shape[0] == 2):
        if(constraint == 'PLANE_STRAIN'):
            C_ = np.array([[C[0,0], C[0,1], 0], [C[1,0], C[1,1], 0], [0, 0, 1]])
    else:
        C_ = C
    
    I3 = np.linalg.det(C_)
    J = np.sqrt(I3)
    I1 = np.trace(C_)
    I2 = 0.5*(np.trace(C_)**2 - np.trace(C_@C_))
    
    return I1, I2, I3, J

# C is full
def get_invariants_iso_np(C, constraint = 'PLANE_STRAIN'):
    I1, I2, I3, J = get_invariants_np(C, constraint)
    return J**(-2/3)*I1, J**(-4/3)*I2, I3, J


# delta E(u;v) : Directional derivative Green-Lagrange from displacements (Mandel)
def get_deltaGL_mandel(u, v):
    conv = ft.get_mechanical_notation_conversor(gdim = u.ufl_shape[0])
    return conv.symgrad_mandel(v) + conv.tensor2mandel(df.grad(u).T*df.grad(v))

# Green-Lagrange from displacements (Mandel)
# Note that E(u) = delta E(0.5*u; u) = eps(u) + 0.5*(grad(u).T*grad(u))
def get_GL_mandel(u):
    return get_deltaGL_mandel(0.5*u, u)

def getF_fromE(E, R):
    U = sp.linalg.sqrtm(2*E + np.eye(E.shape[0]))
    return R@U

def getUmandel_fromEmandel(E):
    E_ = ft.mandel2tensor_np(E)
    return ft.tensor2mandel_np(getF_fromE(E_, np.eye(E_.shape[0])))

