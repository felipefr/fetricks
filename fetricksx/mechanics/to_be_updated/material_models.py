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
import ufl
import fetricks as ft
import numpy as np

def psi_hartmannneff_C(C_, param): # paper german benchmarks Archives, 2021
    
    if(C_.ufl_shape[0] == 2):    
        C = df.as_tensor([[C_[0,0], C_[0,1], 0], [C_[1,0], C_[1,1], 0], [0, 0, 1]])
    else:
        C = C_

    alpha, c10, c01, kappa = param["alpha"], param["c10"], param["c01"], param["kappa"]

    J = df.sqrt(df.det(C))
    Cbar = J**(-2/3)*C
    I1 = df.tr(Cbar)
    I2 = df.tr(ufl.cofac(Cbar))
    # I2 = 0.5*(df.tr(Cbar)**2 - df.tr(Cbar*Cbar))
    
    # print(alpha)
    U = (kappa/50)*(J**5 + J**(-5) - 2)
    # U = 0.5*kappa*(J - 1)**2
    W = alpha*(I1**3 - 27) + c10*(I1 - 3.) + c01*(I2**1.5 - 3.*np.sqrt(3.))
    
    return U + W


def psi_hartmannneff(F, param): # paper german benchmarks Archives, 2021
    return psi_hartmannneff_C(F.T*F, param)


def PK2_hartmannneff_C_np(C_, param):
    if(C_.shape[0] == 2):    
        C = np.array([[C_[0,0], C_[0,1], 0], [C_[1,0], C_[1,1], 0], [0, 0, 1]])
    else:
        C = C_
        
    alpha, c10, c01, kappa = param["alpha"], param["c10"], param["c01"], param["kappa"]

    I3 = np.linalg.det(C)
    I1 = np.trace(C)
    I2 = 0.5*(np.trace(C)**2 - np.trace(C@C))
    
    dpsi1 = 3*I1**2*alpha/I3 + c10/I3**(1/3)
    dpsi2 = 3*c01*(I2/I3**(2/3))**(3/2)/(2*I2)
    dpsi3 = -I1**3*alpha/I3**2 - I1*c10/(3*I3**(4/3)) + kappa*(5*I3**(3/2)/2 - 5/(2*I3**(7/2)))/50 - c01*(I2/I3**(2/3))**(3/2)/I3
    # d2psi11 = 6*I1*alpha/I3
    # d2psi12 = 0
    # d2psi13 = -3*I1**2*alpha/I3**2 - c10/(3*I3**(4/3))
    # d2psi22 = 3*c01*(I2/I3**(2/3))**(3/2)/(4*I2**2)
    # d2psi23 = -3*c01*(I2/I3**(2/3))**(3/2)/(2*I2*I3)
    # d2psi33 = 2*I1**3*alpha/I3**3 + 4*I1*c10/(9*I3**(7/3)) + kappa*(15*sqrt(I3)/4 + 35/(4*I3**(9/2)))/50 + 2*c01*(I2/I3**(2/3))**(3/2)/I3**2
    
    
    a1 = 2*(dpsi1 + I1*dpsi2)
    a2 = -2*dpsi2
    a3 = 2*I3*dpsi3
    
    S = a1*np.eye(C_.shape[0]) + a2*C_ + a3*np.linalg.inv(C_)
    return S


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
    