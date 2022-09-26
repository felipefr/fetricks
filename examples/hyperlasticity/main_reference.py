#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:51:05 2022

@author: felipefr
"""


import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
from fetricks.fenics.mesh.mesh import Mesh 

import fetricks as ft 

from timeit import default_timer as timer
from functools import partial 

df.parameters["form_compiler"]["representation"] = 'uflacs'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

# elastic parameters
def getPsi(e, param):
    tr_e = df.tr(e)
    e2 = df.inner(e, e)

    lamb, mu, alpha = param
    
    return (0.5*lamb*(1.0 + 0.5*alpha*(tr_e**2))*(tr_e**2) + mu*(1 + 0.5*alpha*e2)*(e2))


E = 100.0
nu = 0.3
alpha = 200.0
ty = 5.0


lamb = E*nu/(1+nu)/(1-2*nu)
mu = E/2./(1+nu)

param = [lamb, mu, alpha]

mesh = Mesh("./meshes/mesh_40.xdmf")

start = timer()

clampedBndFlag = 2 
LoadBndFlag = 1 
traction = df.Constant((0.0,ty ))
    
deg_u = 1
deg_stress = 0
V = df.VectorFunctionSpace(mesh, "CG", deg_u)
We = df.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
W = df.FunctionSpace(mesh, We)
W0e = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = df.FunctionSpace(mesh, W0e)

bcL = df.DirichletBC(V, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
bc = [bcL]

def F_ext(v):
    return df.inner(traction, v)*mesh.ds(LoadBndFlag)


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = df.dx(metadata=metadata)

u = df.Function(V, name="Total displacement")
du = df.Function(V, name="Iteration correction")
v = df.TestFunction(V)
u_ = df.TrialFunction(V)


# eps = ft.symgrad(u)
eps_var = df.variable(ft.symgrad(u))
psi = getPsi(eps_var, param)
sig = ft.tensor2mandel(df.diff(psi , eps_var))

# tan = ft.tensor4th2mandel(df.diff(df.diff(psi , eps_var), eps_var))
# tan = df.diff(sig, eps_var)

Pi = psi*dxm - F_ext(u)

# RHS and LHS: Note that Jac = derivative of Res
# a_Newton = df.inner(ft.symgrad_mandel(u_), df.dot(tan, ft.symgrad_mandel(v)) )*dxm
# res = df.inner(ft.symgrad_mandel(v), sig )*dxm - F_ext(v)


res = df.derivative( Pi, u, v)
a_Newton = df.derivative( res, u, u_)

file_results = df.XDMFFile("cook.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


# callbacks = [lambda w, dw: model.update_alpha(ft.tensor2mandel(ft.symgrad(w))) ]

r = ft.Newton(a_Newton, res, bc, du, u , Nitermax = 10, tol = 1e-6)[1]
# ft.Newton_automatic(a_Newton, res, bc, du, u , Nitermax = 10, tol = 1e-8)

## Solve here Newton Raphson

file_results.write(u, 0.0)

end = timer()
print(end - start)

rates = [ np.log(r[i+2]/r[i+1])/np.log(r[i+1]/r[i])  for i in range(len(r) - 2) ] 
print('rates' , rates )