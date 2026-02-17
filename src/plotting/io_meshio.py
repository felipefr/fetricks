#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:10:52 2026

@author: felipe
"""

import numpy as np
import sys
from functools import partial

from petsc4py import PETSc
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

from dolfinx import fem,mesh,plot
import dolfinx.fem.petsc
import ufl
import meshio 


# Geometrie/Mesh
L = 5.0
H = 0.5
Nx = 50
Ny =  5

domain = mesh.create_rectangle(MPI.COMM_WORLD,[[0.0, 0.0],[L,H]],[Nx,Ny],mesh.CellType.triangle, diagonal = mesh.DiagonalType.left)

points = domain.geometry.x
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, 0)
cells = domain.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)

# ---- Build meshio object ----
cell_type_map = {
    1: "line",
    2: "triangle",
    3: "tetra"
}

cell_type = cell_type_map[tdim]
meshio_mesh = meshio.Mesh(
    points=points,
    cells=[(cell_type, cells)]
)


V = fem.functionspace(domain, ("Lagrange", 1, (tdim,)))
S = fem.functionspace(domain, ("DG", 0))
u = fem.Function(V)
s = fem.Function(S)

x = ufl.SpatialCoordinate(domain)
expr_u = ufl.as_vector((0.001*x[0], 0.0))
u.interpolate(fem.Expression(expr_u, V.element.interpolation_points))

def f(x):
    return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])

s.interpolate(f)

# Interpolate to nodal values
u_vals = u.x.array.reshape((-1, tdim))
if(tdim==2):
    u_vals = np.hstack([u_vals, np.zeros((u_vals.shape[0],1))])
    
s_vals = s.x.array

point_data = {"u": u_vals}
cell_data = {"s": [s_vals]}

meshio_mesh.point_data = point_data
meshio_mesh.cell_data = cell_data

# ---- Write ----
meshio.write("mesh.vtu", meshio_mesh)
meshio.write("mesh.xdmf", meshio_mesh)


def f_time(x,t):
    return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])*np.exp(t)

def expr_u_time(t):
    return ufl.as_vector((t*0.001, t*0.01*x[0]))


with meshio.xdmf.TimeSeriesWriter("results_timeseries.xdmf") as writer:
    writer.write_points_cells(points, [(cell_type, cells)])

    for step in range(10):
        s.interpolate(partial(f_time, t = float(step)))
        s_vals = s.x.array
        u.interpolate(fem.Expression(expr_u_time(step), V.element.interpolation_points))
        u_vals[:,0:2] = u.x.array.reshape((-1, tdim))
        
        
        writer.write_data(step, cell_data = cell_data, point_data = point_data)
