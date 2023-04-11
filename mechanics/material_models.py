#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:52:31 2023

@author: ffiguere


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>


Convention: 
    finite strain : F or C in intrisic notation 
    infinitesimal strain: eps in mandel notation

"""


import dolfin as df
import fetricks as ft
import numpy as np
def psi_ciarlet_C(C_, param): # paper german benchmarks Archives, 2021
    if(C_.ufl_shape[0] == 2):    
        C = df.as_tensor([[C_[0,0], C_[0,1], 0], [C_[1,0], C_[1,1], 0], [0, 0, 1]])
    else:
        C = C_

    lamb, mu = param["lamb"], param["mu"]
    I3 = df.det(C)
    I1 = df.tr(C)
    J = df.sqrt(I3)
    
    psi = 0.5*mu*(I1 - 3) + 0.25*lamb*(I3 - 1) - (0.5*lamb + mu)*df.ln(J)
    
    return psi

def psi_ciarlet(F, param): # paper german benchmarks Archives, 2021
    return psi_ciarlet_C(F.T*F, param)


def PK2_ciarlet_C_np(C_, param):
    if(C_.shape[0] == 2):    
        C = np.array([[C_[0,0], C_[0,1], 0], [C_[1,0], C_[1,1], 0], [0, 0, 1]])
    else:
        C = C_
        
    lamb, mu = param["lamb"], param["mu"]
    I3 = np.linalg.det(C)
    
    dpsidi1 = 0.5*mu
    dpsidi3 = 0.25*lamb - 0.5*(0.5*lamb + mu)*I3**(-1)
    # d2psidi3di3 = 1/2*(l/2 + mu)*I3(C)**(-2)
    S = 2*(dpsidi1*np.eye(C_.shape[0]) + dpsidi3*I3*np.linalg.inv(C_))
    return S


def psi_hookean_nonlinear_lame(e, param):
    lamb, mu, alpha = param["lamb"], param["mu"], param["alpha"]
    
    tr_e = ft.tr_mandel(e)
    e2 = df.inner(e, e)

    psi = 0.5*lamb*(1.0 + 0.5*alpha*(tr_e**2))*(tr_e**2) + mu*(1 + 0.5*alpha*e2)*(e2)
    
    return psi



# psi is the SEF
# strain: should match psi convention; 
# strain_ should be a df.variable, isomorph to strain 
def get_stress_tang_from_psi(psi, strain, strain_): 
    stress = df.diff(psi(strain), strain_)
    tang = df.diff(stress, strain_)
    return stress, tang
    