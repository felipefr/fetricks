#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 22:45:49 2022

@author: felipe
"""

from ufl import Jacobian
import dolfin as df

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
