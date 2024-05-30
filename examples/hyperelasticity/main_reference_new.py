#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:16:36 2024

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:16:36 2024

@author: felipefr


This file is part of fetricksx:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricksx: where "fe" stands for FEM, FEniCSx)

Copyright (c) 2023-2024, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or
<f.rocha.felipe@gmail.com>
"""



import os, sys
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import partial 

sys.path.append("/home/felipe/sources/fetricksx")

import numpy as np
from dolfinx import fem, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI


# import fetricksx as ft

# elastic parameters
def getPsi(e, param):
    tr_e = ufl.tr(e)
    e2 = ufl.inner(e, e)

    lamb, mu, alpha = param
    
    return (0.5*lamb*(1.0 + 0.5*alpha*(tr_e**2))*(tr_e**2) + mu*(1 + 0.5*alpha*e2)*(e2))



def solve_hyperlasticity(param):
    
    gdim = param['gdim']

    
    if(param['create_mesh']):
        os.system('gmsh -2 {0} -o {1}'.format(param['geo_file'], param['mesh_file']) )
        
    domain, markers, facets =  io.gmshio.read_from_msh(param['mesh_file'], MPI.COMM_WORLD, gdim = gdim)
    tdim = domain.topology.dim
    fdim = tdim - 1 
    print(tdim, fdim)
    
    lamb = param['E']*param['nu']/(1+param['nu'])/(1-2*param['nu'])
    mu = param['E']/2./(1+param['nu'])
    print(lamb,mu)
    material_param = [fem.Constant(domain,lamb), fem.Constant(domain,mu), fem.Constant(domain,param['alpha'])]
                                                       

    
    # We = ufl.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
    # W = ufl.FunctionSpace(mesh, We)
    # W0e = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    # W0 = ufl.FunctionSpace(mesh, W0e)
    
    

        

         
    Uh = fem.functionspace(domain, ("CG", param['deg_u'], (param['gdim'],)))
    dx = ufl.Measure('dx', domain = domain, subdomain_data=markers, metadata={"quadrature_degree": 4})
    ds = ufl.Measure('ds', domain = domain, subdomain_data=facets)
    u = fem.Function(Uh)
    du = fem.Function(Uh)
    v = ufl.TestFunction(Uh)
    u_ = ufl.TrialFunction(Uh)
    

    bcs_D = []
    for bc in param['dirichlet']:
        bc_dofs = fem.locate_dofs_topological(Uh.sub(bc[1]), fdim, facets.find(bc[0]))
        bcs_D.append(fem.dirichletbc(bc[2], bc_dofs, Uh.sub(bc[1])))
        
    def F_ext(v):
        return sum([ ufl.inner(bc[2], v[bc[1]])*ds(bc[0]) for bc in param['neumann']])

    
    eps_var = ufl.variable(ufl.sym(ufl.grad(u)))
    psi = getPsi(eps_var, material_param)
    # sig = ft.tensor2mandel(ufl.diff(psi , eps_var))
    
    
    Pi = psi*dx - F_ext(u)
    res = ufl.derivative( Pi, u, v)
    jac = ufl.derivative( res, u, u_)
    
    
    problem = fem.petsc.NonlinearProblem(res, u, bcs_D, J = jac)

    solver = dolfinx.nls.petsc.NewtonSolver(domain.comm, problem)
    # Set Newton solver options
    solver.atol = 1e-12
    solver.rtol = 1e-12
    # solver.convergence_criterion = "incremental"
    solver.solve(u)
                                                       
    
    return u

param={
'E': 100.0,
'nu' : 0.3,
'alpha' : 0.0,
'ty' : 5.0,
'clamped_bc' : 4, 
'load_bc' : 2,
'deg_u' : 1,
'deg_stress' : 0, 
'gdim': 2,
'dirichlet': [],
'neumann': [],
'mesh_file' : "./meshes/cook.msh",
'geo_file' : "./meshes/cook.geo",
'create_mesh': True
}

# for dirichlet and neumann: tuple of (physical group tag, direction, value)
param['dirichlet'].append((param['clamped_bc'], 0, 0.))
param['dirichlet'].append((param['clamped_bc'], 1, 0.))
param['neumann'].append((param['load_bc'], 1, param['ty']))

u = solve_hyperlasticity(param)

mesh = u.function_space.mesh

out_file = "hyperelasticity.xdmf"
with io.XDMFFile(MPI.COMM_WORLD, out_file, "w") as xdmf:
    xdmf.write_mesh(mesh)
    
with io.XDMFFile(MPI.COMM_WORLD, out_file, "a") as xdmf:
    xdmf.write_function(u, 0)
    
    
print(np.linalg.norm(u.x.array))
    
# file_results.parameters["flush_output"] = True
# file_results.parameters["functions_share_mesh"] = True


# start = timer()

    






# metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
# dxm = ufl.dx(metadata=metadata)



# # eps = ft.symgrad(u)


# # tan = ft.tensor4th2mandel(ufl.diff(ufl.diff(psi , eps_var), eps_var))
# # tan = ufl.diff(sig, eps_var)


# # RHS and LHS: Note that Jac = derivative of Res
# # a_Newton = ufl.inner(ft.symgrad_mandel(u_), ufl.dot(tan, ft.symgrad_mandel(v)) )*dxm
# # res = ufl.inner(ft.symgrad_mandel(v), sig )*dxm - F_ext(v)


# res = ufl.derivative( Pi, u, v)
# a_Newton = ufl.derivative( res, u, u_)




# # callbacks = [lambda w, dw: model.update_alpha(ft.tensor2mandel(ft.symgrad(w))) ]

# r = ft.Newton(a_Newton, res, bc, du, u , Nitermax = 10, tol = 1e-6)[1]
# # ft.Newton_automatic(a_Newton, res, bc, du, u , Nitermax = 10, tol = 1e-8)

# ## Solve here Newton Raphson

# file_results.write(u, 0.0)

# end = timer()
# print(end - start)

# rates = [ np.log(r[i+2]/r[i+1])/np.log(r[i+1]/r[i])  for i in range(len(r) - 2) ] 
# print('rates' , rates )