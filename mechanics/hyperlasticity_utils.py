#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:56:45 2023

@author: ffiguere
"""

import numpy as np

# Green-Lagrange (E) to Cauchy-Green (C)
# E is mandel
def GL2CG_np(E):
    return 2*E + np.eye(E.shape[0])

# C_ is full
def get_invariants_np(C_):
    
    if(C_.shape[0] == 2):
        C = np.array([[C_[0,0], C_[0,1], 0], [C_[1,0], C_[1,1], 0], [0, 0, 1]])
    else:
        C = C_
    
    I3 = np.linalg.det(C)
    J = np.sqrt(I3)
    I1 = np.trace(C)
    I2 = 0.5*(np.trace(C)**2 - np.trace(C@C))
    
    return I1, I2, I3, J

# C_ is full
def get_invariants_iso_np(C_):
    I1, I2, I3, J = get_invariants_np(C_)
    
    return J**(-2/3)*I1, J**(-4/3)*I2, I3, J
