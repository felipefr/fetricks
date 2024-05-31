#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:11:34 2023

@author: ffiguere

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import numpy.typing as npt
import numpy as np
import dolfinx
from dolfinx import fem
import basix
import ufl

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


def create_normal_contribution_bc(Q: fem.FunctionSpace, expr: ufl.core.expr.Expr,  facets: npt.NDArray[np.int32]) -> fem.Function:
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
    facet_cmap = basix.ufl.element(
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
    normal_expr = fem.Expression(
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
    qh = fem.Function(Q)
    qh._cpp_object.interpolate(
        values.reshape(-1, domain.geometry.dim).T.copy(), boundary_entities[::2].copy())
    qh.x.scatter_forward()
    return qh


def neumannbc(g, flag, V):
    msh = V.mesh    
    fdim = msh.topology.dim - 1
    V_, _ = V.collapse()
    g_ = create_normal_contribution_bc(V_, g*ufl.FacetNormal(msh), msh.facets.find(flag))
    dofs = fem.locate_dofs_topological((V, V_), fdim, msh.facets.find(flag))
    
    return fem.dirichletbc(g_, dofs, V)

def dirichletbc(g, flag, V):
    msh = V.mesh     
    fdim = msh.topology.dim - 1
    dofs = fem.locate_dofs_topological(V, fdim, msh.facets.find(flag))

    return fem.dirichletbc(g, dofs, V)