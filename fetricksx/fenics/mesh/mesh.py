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

"""
Known problems: 
1) self._geometry = self.domain._geometry, ...,  is needed for dolfinx 0.9.0.
   Conversely, it is not needed for 0.8.0
2) 
"""

import os
import dolfinx
from dolfinx import io, mesh
import ufl
from mpi4py import MPI
from functools import reduce
import numpy as np


# import fetricks.fenics.postprocessing.wrapper_io as iofe

class Mesh(mesh.Mesh):
    def __init__(self, meshfile, comm = MPI.COMM_WORLD, gdim = 2):
        if(meshfile[-3:]=='geo'):
            geofile, meshfile = meshfile, meshfile[:-3] + "msh" 
            os.system('gmsh -{0} {1} -o {2}'.format(gdim, geofile, meshfile))
            
        self.domain, self.markers, self.facets = io.gmshio.read_from_msh(meshfile, comm, gdim = gdim)
        
        self._cpp_object = self.domain._cpp_object 
        self._ufl_domain = self.domain._ufl_domain
        
        if(dolfinx.__version__ == '0.9.0'):
            self._ufl_domain._ufl_cargo = self.domain._ufl_domain._ufl_cargo
            self._geometry = self.domain._geometry
            self._topology = self.domain._topology
            
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
         
