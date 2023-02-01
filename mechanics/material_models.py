#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:52:31 2023

@author: ffiguere
"""

import dolfin as df

def psi_ciarlet(F, param): # paper german benchmarks Archives, 2021
    lamb, mu = param["lamb"], param["mu"]
    C = F.T*F
    J2 = df.det(C)
    IC = df.tr(C)
    J = df.sqrt(J2)
    
    psi = 0.5*mu*(IC - 3) + 0.25*lamb*(J2 - 1) - (0.5*lamb + mu)*df.ln(J)
    
    return psi

# psi is the SEF
# strain: should match psi convention; 
# strain_ should be a df.variable, isomorph to strain 
def get_stress_tang_from_psi(psi, strain, strain_): 
    stress = df.diff(psi(strain), strain_)
    tang = df.diff(stress, strain_)
    return stress, tang
    