#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Available in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com,
felipe.figueredorocha@epfl.ch

Bar problem given a constitutive law (single-scale):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the
right. We use an isotropic linear material, given two lamé parameters.
"""
import sys
import dolfin as df
import numpy as np
from ufl import nabla_div, indices
sys.path.insert(0, '../core/')
from fenicsUtils import symgrad

from timeit import default_timer as timer


# Optimization options for the form compiler
df.parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}

resultFolder = './results/'


Lx = 2.0
Ly = 0.5

if(len(sys.argv)>2):
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
else:
    Nx = 100
    Ny = 30
    

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = df.Constant(E/(2*(1 + nu))), df.Constant(E*nu/((1 + nu)*(1 - 2*nu)))

ty = -0.01

# Create mesh and define function space
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                        Nx, Ny, "right/left")

Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# Define boundary condition
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = df.Constant((0.0, ty))

dx = df.Measure('dx', domain=mesh)

start = timer()
# Define functions
duh = df.Function(Uh)            # Incremental displacement
uh_ = df.TrialFunction(Uh)            # Incremental displacement
vh  = df.TestFunction(Uh)             # Test function
uh  = df.Function(Uh)                 # Displacement from previous iteration

# Kinematics
I = df.Identity(2)    # Identity tensor
F = I + df.grad(uh)             # Deformation gradient

C = F.T*F                 # Right Cauchy-Green tensor


# Invariants of deformation tensors
Ic = df.tr(C)
J  = df.det(F)

# Define variational problem
psi = (mu/2)*(Ic - 3) - mu*df.ln(J) + (lmbda/2)*(df.ln(J))**2

Psi = psi*dx - df.inner(traction, uh)*ds 

Res= df.derivative(Psi, uh, vh)
Jac= df.derivative(Res, uh, uh_)

df.solve(Res == 0, uh, bcL, J = Jac) # bug using the jacobian

end = timer()
print(end - start)

# # Save solution in VTK format
# fileResults = df.XDMFFile(resultFolder + "bar_withStress.xdmf")
# fileResults.write_checkpoint(uh, 'u', 0)
# fileResults.close()