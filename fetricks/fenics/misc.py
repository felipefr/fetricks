"""

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import dolfin as df
import numpy as np
from timeit import default_timer as timer


# NON-FLATTENED FUNCTIONS
def symgrad(v): 
    return df.sym(df.grad(v))
   

# Condensed version of the Integral: test
def Integral_shorter(u, dx, shape):
    if(len(shape) == 1):
        return np.array([ df.assemble(u[i]*dx) for i in range(shape[0])]) 

    elif(len(shape) == 2):
        return np.array( [ [ df.assemble(u[i, j]*dx) for j in range(shape[1])] 
                          for i in range(shape[0]) ])

# Vectorial and Tensorial integrals (Fenics integrals are scalars by default)
def Integral(u,dx,shape):
    
    n = len(shape)
    I = np.zeros(shape)
    
    if(type(dx) != type([])):
        dx = [dx]
 
    if(n == 1):
        for i in range(shape[0]):
            for dxj in dx:
                I[i] += df.assemble(u[i]*dxj)
            
    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for dxk in dx:
                    I[i,j] += df.assemble(u[i,j]*dxk)

    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for dxk in dx:
                        I[i,j,k] += df.assemble(u[i,j,k]*dxk)

    elif(n == 4):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        for dxk in dx:
                            I[i,j,k,l] += df.assemble(u[i,j,k,l]*dxk)
    
    else:
        print('not implemented for higher order integral')
        
    
    return I

# Returns a expression affine tranformation x -> a + Bx (given a, B and a mesh)
def affineTransformationExpression(a,B, mesh):
    return df.Expression(('a0 + B00*x[0] + B01*x[1]','a1 + B10*x[0] + B11*x[1]'), a0 = a[0], a1 = a[1],
               B00=B[0,0], B01 = B[0,1], B10 = B[1,0], B11= B[1,1] ,degree = 1, domain = mesh)

# Returns a expression (x[0], x[1])
def VecX_expression(degree = 1):
    return df.Expression(('x[0]','x[1]'), degree = degree)


# Used to implement the Piola transofmation
class myfog(df.UserExpression): # fog f,g : R2 -> R2, generalise 
    def __init__(self, f, g, **kwargs):
        self.f = f 
        self.g = g
        f.set_allow_extrapolation(True)
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[:2] = self.f(self.g(x))
        
    def value_shape(self):
        return (2,)

# Used to implement the Piola transofmation
class myfog_expression(df.UserExpression): # fog f,g : R2 -> R2, generalise 
    def __init__(self, f, g, **kwargs):
        self.f = f 
        self.g = g
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[:2] = self.f(self.g(x))
        
    def value_shape(self):
        return (2,)
    
    

# create quadrature spaces: scalar, vectorial  (strain/stresses), and tensorial (tangents)
def create_quadrature_spaces_mechanics(mesh, deg_q, qdim):
    cell = mesh.ufl_cell()
    q = "Quadrature"
    QF = df.FiniteElement(q, cell, deg_q, quad_scheme="default")
    QV = df.VectorElement(q, cell, deg_q, quad_scheme="default", dim=qdim)
    QT = df.TensorElement(q, cell, deg_q, quad_scheme="default", shape=(qdim, qdim))
    return [df.FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]


def create_DG_spaces_mechanics(mesh, deg_q, qdim):
    cell = mesh.ufl_cell()
    q = "DG"
    QF = df.FiniteElement(q, cell, deg_q)
    QV = df.VectorElement(q, cell, deg_q, dim=qdim)
    QT = df.TensorElement(q, cell, deg_q, shape=(qdim, qdim))
    return [df.FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]


# apparently add_local is faster than set_local
def setter(q, values):
    """
    q:
        quadrature function space
    values:
        entries for `q`
    """
    v = q.vector()
    v.zero()
    v.add_local(values.flatten())
    v.apply("insert")


