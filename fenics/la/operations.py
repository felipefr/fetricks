#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:25:53 2023

@author: ffiguere

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import ufl
import fetricks as ft

# Cijkl = Aik Bjl
def outer_overline_ufl(A, B):
    i, j, k, l = ufl.indices(4)
    return ufl.as_tensor(A[i,k]*B[j,l], (i,j,k,l))

# Cijkl = Ail Bjk
def outer_underline_ufl(A, B):
    i, j, k, l = ufl.indices(4)
    return ufl.as_tensor(A[i,l]*B[j,k], (i,j,k,l))

# Product defined in [Holzapfel, p. 254, eq. 165].
def outer_dot_ufl(A,B):
    return 0.5*(outer_overline_ufl(A,B) + outer_underline_ufl(A,B))

def outer_dot_mandel_ufl(A, B, conv = None):
    if(not conv):
        conv = ft.get_mechanical_notation_conversor(dim_strain = A.ufl_shape[0])
    return conv.tensor4th2mandel(outer_dot_ufl(conv.mandel2tensor(A), conv.mandel2tensor(B))) 