#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 21:15:04 2026

@author: felipe
"""

import numpy as np
import sys
from functools import partial

from mpi4py import MPI

from dolfinx import fem,mesh,plot, io
import ufl
import meshio 

import fetricksx as ft

# Geometrie/Mesh
outfile = "results.xdmf"
L = 5.0
H = 0.5
Nx = 50
Ny =  5

domain = mesh.create_rectangle(MPI.COMM_WORLD,[[0.0, 0.0],[L,H]],[Nx,Ny],mesh.CellType.triangle, diagonal = mesh.DiagonalType.left)

V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
S = fem.functionspace(domain, ("DG", 0))
u = fem.Function(V, name = 'u')
s = fem.Function(S, name = 's')

x = ufl.SpatialCoordinate(domain)

def f_time(x,t):
    return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])*np.exp(t)

t_ = fem.Constant(domain, 0.0)
expr_u = fem.Expression(ufl.as_vector((t_*0.001, t_*0.01*x[0])) , V.element.interpolation_points, comm = MPI.COMM_WORLD)

with ft.XDMFWriter(domain, outfile) as f:
    f.write_mesh()
    f.register_field(u, export_dim = 3)
    f.register_field(s)
    for step in range(10):
        s.interpolate(partial(f_time, t = float(step)))
        t_.value = float(step)
        u.interpolate(expr_u)
        f.write_fields(step)

with io.XDMFFile(MPI.COMM_WORLD, outfile, "r") as xdmf:
    domain_read = xdmf.read_mesh()

Vread = fem.functionspace(domain_read, ("Lagrange", 1, (3,)))
Sread = fem.functionspace(domain_read, ("DG", 0))
uread = fem.Function(Vread, name = "u")
sread = fem.Function(Sread, name = "s")

with ft.XDMFReader(outfile) as f:               
    f.read_field(uread, 9)
    f.read_field(sread, 9)
    
assert np.allclose(sread.x.array, s.x.array)
assert np.allclose(u.x.array, uread.x.array.reshape((-1,3))[:,:2].flatten())