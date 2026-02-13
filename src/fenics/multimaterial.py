#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 20:17:35 2022

@author: felipe


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import numpy as np

def create_piecewise_constant_field(domain, cell_markers, property_dict, name=None):
    "Copied from https://bleyerj.github.io/comet-fenicsx/tips/piecewise_constant_field/piecewise_constant_field.html"
    """Create a piecewise constant field with different values per subdomain.

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    cell_markers : MeshTag
        cell marker MeshTag
    property_dict : dict
        A dictionary mapping region tags to physical values {tag: value}

    Returns
    -------
    A DG-0 function
    """
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k


# def getLameExpression(nu1,E1,nu2,E2,M, op= 'cpp', plane_stress = False ):
    
#     eng2lamb_ = eng2lambPlaneStress if plane_stress else eng2lamb
    
#     mu1 = eng2mu(nu1,E1)
#     lamb1 = eng2lamb_(nu1,E1)
#     mu2 = eng2mu(nu2,E2)
#     lamb2 = eng2lamb_(nu2,E2)

#     param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1,mu1], [lamb2,mu2]])
    
#     return getMultimaterialExpression(param, M, op = op)