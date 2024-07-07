#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:36:23 2024

@author: felipe
"""


"""

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

from dolfinx import io, mesh
import ufl
from mpi4py import MPI
from functools import reduce
import numpy as np

# import fetricks.fenics.postprocessing.wrapper_io as iofe

class Mesh(mesh.Mesh):
    def __init__(self, meshfile, comm = MPI.COMM_WORLD, gdim = 2):
        self.domain, self.markers, self.facets = io.gmshio.read_from_msh(meshfile, comm, gdim = gdim)
        self._cpp_object = self.domain._cpp_object 
        self._ufl_domain = self.domain._ufl_domain
        self._ufl_domain._ufl_cargo = self.domain._ufl_domain._ufl_cargo
        self.createMeasures()
        self.gdim = self.domain.geometry.dim
        self.tdim = self.domain.topology.dim
        self.num_cells = len(self.domain.topology.connectivity(self.tdim,0))

        # self.vols = np.array([df.Cell(self, i).volume() for i in range(self.num_cells())])
        self.dsN = {}
        self.dxR = {}

    def boundaries(self):
        return self.facets
    
    def subdomains(self):
        return self.markers
    
    def createMeasures(self):
         self.ds = ufl.Measure('ds', domain=self, subdomain_data=self.facets)
         self.dx = ufl.Measure('dx', domain=self, subdomain_data=self.markers)
         
