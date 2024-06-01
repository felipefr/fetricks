#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 12:13:13 2024

@author: felipefr
"""

from basix.ufl import element, mixed_element
from dolfinx import fem

# fe_params : tuple family, degree
def mixed_functionspace(msh, fe_params): 
    FEs = [element(fe[0], msh.domain.basix_cell(), fe[1]) for fe in fe_params]
    Wh = fem.functionspace(msh, mixed_element(FEs))
    return Wh