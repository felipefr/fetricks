#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 22:45:49 2022

@author: felipe
"""

from ufl import Jacobian
import dolfin as df
import fetricks as ft
from functools import partial

# def tgrad(u, t):
#     return df.grad(df.dot(u,t)) # dot(grad(u).T, t)  

def grad_truss(u, t):
    return df.dot(df.grad(df.dot(u,t)), t) # inner( outer(t,t) , grad(u)) 


def get_mesh_truss(X, cells):    
    mesh = df.Mesh()
    ed = df.MeshEditor()
    ed.open(mesh, "interval", 1, 2)
    ed.init_vertices( len(X))
    for i, v in enumerate(X):
        print(i)
        ed.add_vertex(i, v)

    ed.init_cells(len(cells))
    for i, c in enumerate(cells):
        print(i)
        ed.add_cell(i, c)    
        
    ed.close()
    
    return mesh

def get_tangent_truss(mesh):
    J = Jacobian(mesh)
    se = df.VectorElement("DG", mesh.ufl_cell(), 0, dim = 2)
    sh0 = df.FunctionSpace(mesh, se)
    t = df.as_vector([J[0, 0], J[1, 0]])/df.sqrt(df.inner(J,J))
    
    return df.project(t, sh0)


def flag_point(p, gdim): 
    foo = lambda x, on_boundary: all([df.near(x[i], p[i]) for i in range(gdim)])
    return foo

def posproc_truss(param, uh):
    mesh = uh.function_space().mesh()

    se = df.VectorElement("DG", mesh.ufl_cell(), 0, dim = 1)
    sh0 = df.FunctionSpace(mesh, se)
    
    
    dx = df.Measure('dx', mesh)
    t = get_tangent_truss(mesh)
    grad_truss = partial(ft.grad_truss, t = t)
    sigma_law = param['sigma_law']
        
    strain = grad_truss(uh)
    stress = sigma_law(strain)
    
    strain_h = ft.QuadratureFunction(sh0, dxm = dx, name = 'strain')
    strain_h.update(df.as_vector((strain,)))

    stress_h = ft.QuadratureFunction(sh0, dxm = dx, name = 'stress')
    stress_h.update(df.as_vector((stress,)))

    return {'strain': strain_h, 'stress': stress_h}

def solve_truss(param, mesh = None):
    if(not mesh):
        mesh = get_mesh_truss(param['X'], param['cells'])
    gdim = param['X'].shape[1]
    Area = param['Area']
    sigma_law = param['sigma_law']
    
    t = get_tangent_truss(mesh)
    
    grad_truss = partial(ft.grad_truss, t = t)
    
    Ue = df.VectorElement("CG", mesh.ufl_cell(), 1, dim=gdim)
    Uh = df.FunctionSpace(mesh, Ue)    
    
    bcs_D = [ df.DirichletBC(Uh.sub(bc[1]), df.Constant(bc[2]), flag_point(bc[0], gdim), method = 'pointwise') for bc in param['dirichlet'] ]
    bcs_N = [ df.PointSource(Uh.sub(bc[1]), bc[0], bc[2]) for bc in param['neumann']]
    
    # # Define variational problem
    uh = df.TrialFunction(Uh) 
    vh = df.TestFunction(Uh)
    
    dx = df.Measure('dx', mesh)
    
    a = df.inner( Area*sigma_law(grad_truss(uh)),  grad_truss(vh))*dx
    b = df.inner( df.Constant((0.,0.)) , vh)*dx # no body forces
    
    A, b = df.assemble_system(a, b, bcs_D)
    
    for bc in bcs_N: 
        bc.apply(b) # Only in b (I don't why)

    uh = df.Function(Uh)
    
    # Compute solution
    df.solve(A, uh.vector(), b)   
        
    return uh

