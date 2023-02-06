#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:52:31 2023

@author: ffiguere

Convention: 
    finite strain : F in intrisic notation 
    infinitesimal strain: eps in mandel notation

"""


import dolfin as df
import fetricks as ft

def psi_ciarlet(F, param): # paper german benchmarks Archives, 2021
    lamb, mu = param["lamb"], param["mu"]
    C = F.T*F
    J2 = df.det(C)
    IC = df.tr(C)
    J = df.sqrt(J2)
    
    psi = 0.5*mu*(IC - 3) + 0.25*lamb*(J2 - 1) - (0.5*lamb + mu)*df.ln(J)
    
    return psi


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
    