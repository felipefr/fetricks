#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:30:11 2024

@author: felipe
"""

"""
Compute DirichletBC in normal direction on arbitrary domains
Author: Jørgen S. Dokken
Licenses:
- LGPL: 3.0 for `compute_exterior_facet_entities`
- All other code is licensed under MIT
"""


import numpy.typing as npt
from mpi4py import MPI
import dolfinx
from basix.ufl import element, mixed_element
from dolfinx import fem, io, mesh, fem
import ufl
import numpy as np


def compute_exterior_facet_entities(mesh, facets):
    """Helper function to compute (cell, local_facet_index) pairs for exterior facets
    Licensed under LGPL: 3.0, part of DOFLINx, https://github.com/FEniCS/dolfinx/
    """
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim - 1)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    integration_entities = np.empty(2 * len(facets), dtype=np.int32)
    for i, facet in enumerate(facets):
        cells = f_to_c.links(facet)
        assert len(cells) == 1
        cell = cells[0]
        local_facets = c_to_f.links(cell)
        local_pos = np.flatnonzero(local_facets == facet)
        integration_entities[2 * i] = cell
        integration_entities[2 * i + 1] = local_pos[0]
    return integration_entities


def create_normal_contribution_bc(Q: dolfinx.fem.FunctionSpace, expr: ufl.core.expr.Expr,  facets: npt.NDArray[np.int32]) -> dolfinx.fem.Function:
    """
    Create function representing normal vector
    SPDX-License-Identifier:    MIT
    Author: Jørgen S. Dokken
    """
    domain = Q.mesh
    Q_el = Q.element
    # Compute integration entities (cell, local_facet index) for all facets
    boundary_entities = compute_exterior_facet_entities(domain, facets)
    interpolation_points = Q_el.basix_element.x
    fdim = domain.topology.dim - 1

    c_el = domain.ufl_domain().ufl_coordinate_element()
    ref_top = c_el.reference_topology
    ref_geom = c_el.reference_geometry

    cell_to_facet = {"interval": "vertex",
                     "triangle": "interval", "quadrilateral": "interval",
                     "tetrahedron": "triangle", "hexahedron": "quadrilateral"}
    # Pull back interpolation points from reference coordinate element to facet reference element
    facet_cmap = element(
        "Lagrange", cell_to_facet[domain.topology.cell_name()], c_el.degree, shape=(domain.geometry.dim, ), dtype=np.float64)
    facet_cel = dolfinx.cpp.fem.CoordinateElement_float64(
        facet_cmap.basix_element._e)
    reference_facet_points = None
    for i, points in enumerate(interpolation_points[fdim]):
        geom = ref_geom[ref_top[fdim][i]]
        ref_points = facet_cel.pull_back(points, geom)
        # Assert that interpolation points are all equal on all facets
        if reference_facet_points is None:
            reference_facet_points = ref_points
        else:
            assert np.allclose(reference_facet_points, ref_points)

    # Create expression for BC
    normal_expr = dolfinx.fem.Expression(
        expr, reference_facet_points)

    points_per_entity = [sum(ip.shape[0] for ip in ips)
                         for ips in interpolation_points]
    offsets = np.zeros(domain.topology.dim+2, dtype=np.int32)
    offsets[1:] = np.cumsum(points_per_entity[:domain.topology.dim+1])
    values_per_entity = np.zeros(
        (offsets[-1], domain.geometry.dim), dtype=dolfinx.default_scalar_type)
    entities = boundary_entities.reshape(-1, 2)
    values = np.zeros(entities.shape[0]*offsets[-1]*domain.geometry.dim)
    for i, entity in enumerate(entities):
        insert_pos = offsets[fdim] + \
            reference_facet_points.shape[0] * entity[1]
        normal_on_facet = normal_expr.eval(domain, entity)
        values_per_entity[insert_pos: insert_pos + reference_facet_points.shape[0]
                          ] = normal_on_facet.reshape(-1, domain.geometry.dim)
        values[i*offsets[-1] *
               domain.geometry.dim: (i+1)*offsets[-1]*domain.geometry.dim] = values_per_entity.reshape(-1)
    qh = dolfinx.fem.Function(Q)
    qh._cpp_object.interpolate(
        values.reshape(-1, domain.geometry.dim).T.copy(), boundary_entities[::2].copy())
    qh.x.scatter_forward()
    return qh


domain = mesh.create_unit_square(
    MPI.COMM_WORLD, 13, 15, mesh.CellType.quadrilateral)

k = 2
Q_el = element("BDMCF", domain.basix_cell(), k)
P_el = element("DG", domain.basix_cell(), k - 1)
V_el = mixed_element([Q_el, P_el])
V = fem.functionspace(domain, V_el)
Q, _ = V.sub(0).collapse()

fdim = domain.topology.dim - 1
facets_top = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], 1.0))
facets_bottom = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], 0.0))
vals_top = np.full_like(facets_top, 3, dtype=np.int32)
vals_bottom = np.full_like(facets_bottom, 7, dtype=np.int32)
all_facets = np.hstack([facets_top, facets_bottom])
sort = np.argsort(all_facets)
all_values = np.hstack([vals_top, vals_bottom])[sort]
mt = mesh.meshtags(domain, fdim, all_facets[sort], all_values)


dofs_bottom = fem.locate_dofs_topological((V.sub(0), Q), fdim, mt.find(7))


expr = ufl.FacetNormal(domain)
nh_top = create_normal_contribution_bc(Q, expr,  mt.find(3))

Q, _ = V.sub(0).collapse()
dofs_top = fem.locate_dofs_topological((V.sub(0), Q), fdim, mt.find(3))
bc_top = fem.dirichletbc(nh_top, dofs_top, V.sub(0))

nh_bot = create_normal_contribution_bc(Q, expr, mt.find(7))
bc_bottom = fem.dirichletbc(nh_bot, dofs_bottom, V.sub(0))


bcs = [bc_top, bc_bottom]
w_h = dolfinx.fem.Function(V)
dolfinx.fem.set_bc(w_h.x.array, bcs)
w_h.x.scatter_forward()
sigma_h, u_h = w_h.split()


Vl = fem.functionspace(domain, ("DG", k+1, (domain.geometry.dim,)))
sigmal_h = fem.Function(Vl)
sigmal_h.interpolate(sigma_h)

n_out = fem.Function(Vl)
n_out.name = "nh_top"
n_out.interpolate(nh_top)
n_b = fem.Function(Vl)
n_b.name = "nh_bottom"
n_b.interpolate(nh_bot)

with io.VTXWriter(domain.comm, "test.bp", [sigmal_h, n_out, n_b], engine="BP4") as file:
    file.write(0)