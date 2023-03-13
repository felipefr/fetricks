"""

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import fetricks as ft 

from timeit import default_timer as timer
from functools import partial 

df.parameters["form_compiler"]["representation"] = 'uflacs'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

# elastic parameters

E = 100.0
nu = 0.3
alpha = 200.0
ty = 5.0


mesh = ft.Mesh("./meshes/mesh_40.xdmf")

start = timer()

clampedBndFlag = 2 
LoadBndFlag = 1 
traction = df.Constant((0.0,ty ))
    
deg_u = 1
deg_stress = 0
dim_strain = 3
V = df.VectorFunctionSpace(mesh, "CG", deg_u)

bcL = df.DirichletBC(V, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
bc = [bcL]

def F_ext(v):
    return df.inner(traction, v)*mesh.ds(LoadBndFlag)


model = ft.hyperelasticModel(mesh, {'E': E, 'nu': nu, 'alpha': alpha}, deg_stress, dim_strain)
dxm = model.dxm

u = df.Function(V, name="Total displacement")
du = df.Function(V, name="Iteration correction")
v = df.TestFunction(V)
u_ = df.TrialFunction(V)

# RHS and LHS: Note that Jac = derivative of Res
a_Newton = df.inner(ft.tensor2mandel(ft.symgrad(u_)), model.tangent_op(ft.tensor2mandel(ft.symgrad(v))) )*dxm
res = df.inner(ft.tensor2mandel(ft.symgrad(v)), model.stress )*dxm - F_ext(v)

file_results = df.XDMFFile("cook.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


callbacks = [lambda w, dw: model.update(ft.tensor2mandel(ft.symgrad(w))) ]

r = ft.Newton(a_Newton, res, bc, du, u, callbacks , Nitermax = 10, tol = 1e-6)[1]

## Solve here Newton Raphson

file_results.write(u, 0.0)

end = timer()

print(end - start)

rates = [ np.log(r[i+2]/r[i+1])/np.log(r[i+1]/r[i])  for i in range(len(r) - 2) ] 
print('rates' , rates )