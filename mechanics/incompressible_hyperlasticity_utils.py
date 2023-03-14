#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:35:16 2023

@author: ffiguere


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>

"""

import dolfin as df
import fetricks as ft


# get Dvol [Holzapfel, p. 254, eq 6.166]
def getSvol(p, Cinv, J):
    return p*J*Cinv 

# get Dvol [Holzapfel, p. 254, eq 6.166]  (in Mandel convention)
def getDvol(p, Cinv, J):
    Cinv_circ_dot_Cinv = ft.outer_dot_mandel_ufl(Cinv,Cinv)
    Cinv_outer_Cinv = df.outer(Cinv, Cinv) 
    return J*p*(-2*Cinv_circ_dot_Cinv + Cinv_outer_Cinv)

# Deviatoric for S (page 230 Holzapfel)
def Dev(Sbar,C,Cinv):
    return Sbar - (1./3.)*df.inner(Sbar,C)*Cinv

# get Siso given Sbar (page 230 Holzapfel)
def getSiso(Sbar, C, Cinv, J):
    return J**(-2/3)*Dev(Sbar,C,Cinv)

# get Diso given Sbar, Dbar (page 255 Holzapfel). Mandel notation convention
def getDiso(Dbar_, Sbar, C, Cinv, J):
    
    Dbar = J**(-4/3)*Dbar_
    
    Siso = getSiso(Sbar, C, Cinv, J)
    TrCoeff = (2/3)*J**(-2/3)*df.inner(Sbar,C)
    b = df.dot(Dbar, C)
    CDbarC = df.inner(C, b)
    B = (1/3)*Cinv
    
    gamma = CDbarC - 3*TrCoeff
    c = b + 2*Siso
    
    Cinv_circ_dot_Cinv = ft.outer_dot_mandel_ufl(Cinv, Cinv)
    
    return Dbar - 2*df.sym(df.outer(B, c)) + gamma*df.outer(B,B) +  TrCoeff*(Cinv_circ_dot_Cinv)

# More intuitive, but there's probably a bug
# get Diso given Sbar, Dbar (page 255 Holzapfel). Mandel notation convention
# def getDiso4Dbar(Dbar_, Sbar, C, Cinv, J):
    
#     Dbar = J**(-4/3)*Dbar_
    
#     i, j, k, l = ufl.indices(4)
    
#     Siso = getSiso4Sbar(Sbar, C, Cinv, J)
#     TrCoeff = (2/3)*J**(-2/3)*df.inner(Sbar,C)
#     DbarC = df.dot(Dbar, C)
#     CDbarC = df.inner(C, DbarC)
    
#     Cinv_ = conv.mandel2tensor(Cinv)
#     Cinv_circ_dot_Cinv = 0.5*tensor4th2mandel(ufl.as_tensor(Cinv_[i,k]*Cinv_[j,l] + Cinv_[i,l]*Cinv_[j,k] , (i,j,k,l) ))
    
#     return ( Dbar - (2./3.)*df.sym(df.outer(DbarC, Cinv)) + ((1./9.)*CDbarC - (1/3)*TrCoeff)*df.outer(Cinv,Cinv) + 
#               TrCoeff*(Cinv_circ_dot_Cinv) - (2./3.)*df.sym(df.outer(Siso, Cinv)) )