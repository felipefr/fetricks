#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:18:10 2021

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

# lexographic notation : extension of voigt for (00,01,10,11)
Id_lex_df = ufl.as_vector([1.0, 0.0, 0.0, 1.0])
Id_lex_np = np.array([1.0, 0.0, 0.0, 1.0])

def lex2tensor_list(X):
    return [[X[0], X[1]],
            [X[2], X[4]]]

def tensor2lex_list(X):
     return [X[0,0], X[0,1], X[1,0], X[1,1]]

def lex2tensor_np(X):
    return np.array(lex2tensor_list(X))

def tensor2lex_np(X):
    return np.array(tensor2lex_list(X))

def lex2tensor(X):
    return ufl.as_tensor(lex2tensor_list(X))

def tensor2lex(X):
     return ufl.as_vector(tensor2lex_list(X))

def tensor4th2lex_list(X):
    return [[X[0,0,0,0], X[0,0,0,1], X[0,0,1,0], X[0,0,1,1]],
            [X[0,1,0,0], X[0,1,0,1], X[0,1,1,0], X[0,1,1,1]],
            [X[1,0,0,0], X[1,0,0,1], X[1,0,1,0], X[1,0,1,1]],
            [X[1,1,0,0], X[1,1,0,1], X[1,1,1,0], X[1,0,1,1]]]

def tensor4th2lex(X):
    return ufl.as_tensor(tensor4th2lex_list(X))

def tensor4th2lex_np(X):
    return np.array(tensor4th2lex_list(X))

def tr_lex(X):
    return X[0] + X[3]

def grad_lex(v): # it was shown somehow to have better performance than doing it explicity
    return ufl.as_vector([v[0].dx(0), v[0].dx(1), v[1].dx(0), v[1].dx(1)])
    
def macro_strain_lex(i): 
    Eps_unsym = np.zeros((4,))
    Eps_unsym[i] = 1
    return lex2tensor_np(Eps_unsym)

# Unsymmetric notation : extension of voigt for (00,11,01,10)
Id_unsym_df = ufl.as_vector([1.0, 1.0, 0.0, 0.0])
Id_unsym_np = np.array([1.0, 1.0, 0.0, 0.0])

def unsym2tensor_list(X):
    return [[X[0], X[2]],
            [X[3], X[1]]]

def tensor2unsym_list(X):
     return [X[0,0], X[1,1], X[0,1], X[1,0]]

def unsym2tensor_np(X):
    return np.array(unsym2tensor_list(X))

def tensor2unsym_np(X):
    return np.array(tensor2unsym_list(X))

def unsym2tensor(X):
    return ufl.as_tensor(unsym2tensor_list(X))

def tensor2unsym(X):
     return ufl.as_vector(tensor2unsym_list(X))

def tensor4th2unsym_list(X):
    return [[X[0,0,0,0], X[0,0,1,1], X[0,0,0,1], X[0,0,1,0]],
            [X[1,1,0,0], X[1,1,1,1], X[1,1,0,1], X[1,1,1,0]],
            [X[0,1,0,0], X[0,1,1,1], X[0,1,0,1], X[0,1,1,0]],
            [X[1,0,0,0], X[1,0,1,1], X[1,0,0,1], X[1,0,1,0]]]

def tensor4th2unsym(X):
    return ufl.as_tensor(tensor4th2unsym_list(X))

def tensor4th2unsym_np(X):
    return np.array(tensor4th2unsym_list(X))

def tr_unsym(X):
    return X[0] + X[1]

def grad_unsym(v): # it was shown somehow to have better performance than doing it explicity
    return ufl.as_vector([v[0].dx(0), v[1].dx(1), v[0].dx(1), v[1].dx(0)])
    
def macro_strain_unsym(i): 
    Eps_unsym = np.zeros((4,))
    Eps_unsym[i] = 1
    return unsym2tensor_np(Eps_unsym)

# MANDEL NOTATION RELATED FUNCTIONS

sqrt2 = np.sqrt(2)
halfsqrt2 = 0.5*np.sqrt(2)

Id_mandel_df = ufl.as_vector([1.0, 1.0, 0.0])
Id_mandel_np = np.array([1.0, 1.0, 0.0])

def mandel2tensor_np(X):
    return np.array([[X[0], halfsqrt2*X[2]],
                     [halfsqrt2*X[2], X[1]]])

def tensor2mandel_np(X):
    return np.array([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])



def tensor2mandel(X):
    return ufl.as_vector([X[0,0], X[1,1], halfsqrt2*(X[0,1] + X[1,0])])


def mandel2tensor(X):
    return ufl.as_tensor([[X[0], halfsqrt2*X[2]],
                        [halfsqrt2*X[2], X[1]]])

def tensor4th2mandel(X):
    return ufl.as_tensor([ [X[0,0,0,0], X[0,0,1,1], sqrt2*X[0,0,0,1]],
                          [X[1,1,0,0], X[1,1,1,1], sqrt2*X[1,1,0,1]],
                          [sqrt2*X[0,1,0,0], sqrt2*X[0,1,1,1], 2*X[0,1,0,1]] ] )
                      

def tensor4th2mandel_np(X):
    return np.array([ [X[0,0,0,0], X[0,0,1,1], sqrt2*X[0,0,0,1]],
                          [X[1,1,0,0], X[1,1,1,1], sqrt2*X[1,1,0,1]],
                          [sqrt2*X[0,1,0,0], sqrt2*X[0,1,1,1], 2*X[0,1,0,1]] ] )

def tr_mandel(X):
    return X[0] + X[1]


def symgrad_mandel(v): # it was shown somehow to have better performance than doing it explicity
    return ufl.as_vector([v[0].dx(0), v[1].dx(1), halfsqrt2*(v[0].dx(1) + v[1].dx(0))])
    

# Used to convert dPsi/deps_m in mandel notation to the correct stress in mandel notation  
def grad2mandel_vec(X):
    return ufl.as_tensor([X[0] , X[1], 2*X[2]])

# Used to convert dsigma_m/deps_m in mandel notation to the correct tangent tensor in mandel notation  
def grad2mandel_ten(X):
    return ufl.as_tensor([ [X[0,0] , X[0,1], 2*X[0,2]],
                          [X[1,0] , X[1,1], 2*X[1,2]],
                          [X[2,0] , X[2,1], 4*X[2,2]] ])
    
# derive in tensor format and convert to mandel format (scalar)
def mandelgrad(f, x):
    return tensor2mandel(ufl.diff(f,x))

# derive in tensor format and convert to mandel format (2-tensor)
def mandelgrad_ten(f, x):
    return tensor4th2mandel(ufl.diff(f,x))
    
# this is in mandel
def macro_strain_mandel(i): 
    Eps_Mandel = np.zeros((3,))
    Eps_Mandel[i] = 1
    return mandel2tensor_np(Eps_Mandel)



# STRESS RELATED FUNCTIONS
def sigmaLame(u, lame):
    return lame[0]*ufl.div(u)*ufl.Identity(2) + 2*lame[1]*symgrad(u)

def vonMises(sig):
    s = sig - (1./3)*ufl.tr(sig)*ufl.Identity(2)
    return ufl.sqrt((3./2)*ufl.inner(s, s)) 


# mandel to voigt conversions
def mandel2voigtStrain(v, backend = ufl.as_vector):
    return backend([v[0], v[1], sqrt2*v[2]]) 
                    
def mandel2voigtStress(v, backend = ufl.as_vector):
    return backend([v[0], v[1], halfsqrt2*v[2]]) 


# Q = [[np.cos(theta), np.sin(theta)],
#      [-np.sin(theta), np.cos(theta)],
# Tm = [ [Q[0,0]**2 , Q[0,1]**2, sq2*Q[0,0]*Q[0,1]], 
#      [Q[1,0]**2 , Q[1,1]**2, sq2*Q[1,1]*Q[1,0]],
#      [sq2*Q[1,0]*Q[0,0] , sq2*Q[0,1]*Q[1,1], Q[1,1]*Q[0,0] + Q[0,1]*Q[1,0] ]]
# (Q.T @ A @ Q)_m = Tm @ Am
def rotation_mandel(theta):

    c = np.cos(theta)
    s = np.sin(theta)
    c2 = c*c
    s2 = s*s
    cs = c*s
    sq2 = np.sqrt(2.0)
    
    # Rotation tranformation in mandel-kelvin convention
    
    Tm = np.array([ [c2 , s2, -sq2*cs], 
                    [s2 , c2,  sq2*cs],
                    [sq2*cs , -sq2*cs, c2 - s2] ])
    
    
    return Tm
