#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:18:10 2021

@author: felipefr
"""
import dolfin as df
import numpy as np



def local_project(v,V, metadata = {}):
    M = V.mesh()
    dv = df.TrialFunction(V)
    v_ = df.TestFunction(V)
    dx = df.Measure('dx', M, metadata = metadata)
    a_proj = df.inner(dv,v_)*dx 
    b_proj = df.inner(v,v_)*dx
    solver = df.LocalSolver(a_proj,b_proj) 
    solver.factorize()
    u = df.Function(V)
    solver.solve_local_rhs(u)
    return u

def symgrad(v): return df.sym(df.grad(v))


def symgrad_voigt(v):
    return df.as_vector([v[0].dx(0), v[1].dx(1), v[0].dx(1) + v[1].dx(0)])


def macro_strain(i):
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.],
                    [Eps_Voigt[2]/2., Eps_Voigt[1]]])

def stress2Voigt(s, backend = df.as_vector):
    return backend([s[0, 0], s[1, 1], s[0, 1]])


def strain2Voigt(e, backend = df.as_vector):
    return backend([e[0, 0], e[1, 1], 2*e[0, 1]])

def voigt2Strain(e, backend = df.as_vector):
    return backend([[e[0], 0.5*e[2]], [0.5*e[2], e[1]]])

def voigt2Stress(s, backend = df.as_vector):
    return backend([[s[0], s[2]], [s[2], s[1]]])

def Integral(u, dx, shape):
    n = len(shape)
    valueIntegral = np.zeros(shape)

    if(n == 1):
        for i in range(shape[0]):
            valueIntegral[i] = df.assemble(u[i]*dx)

    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                valueIntegral[i, j] = df.assemble(u[i, j]*dx)

    return valueIntegral
