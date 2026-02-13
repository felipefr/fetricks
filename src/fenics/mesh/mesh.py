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
1) self._geometry = self.mesh._geometry, ...,  is needed for dolfinx 0.9.0.
   Conversely, it is not needed for 0.8.0
2) 
"""

import os
import dolfinx
from dolfinx import io, mesh
import ufl
from mpi4py import MPI

class Mesh(mesh.Mesh):
    def __init__(self, meshfile, comm = MPI.COMM_WORLD, gdim = 3):
        if(meshfile[-3:]=='geo'):
            geofile, meshfile = meshfile, meshfile[:-3] + "msh" 
            os.system('gmsh -{0} {1} -o {2}'.format(gdim, geofile, meshfile))
        
        
        print(meshfile)
        meshdata = io.gmsh.read_from_msh(meshfile, comm, gdim = gdim)
        self.mesh = meshdata.mesh
        self.cell_tags = meshdata.cell_tags 
        self.facet_tags = meshdata.facet_tags
        self.physical = meshdata.physical_groups
        
        self._cpp_object = self.mesh._cpp_object 
        self._ufl_domain = self.mesh._ufl_domain
        
        if(dolfinx.__version__ == '0.9.0' or '0.10.0'):
            self._ufl_domain._ufl_cargo = self.mesh._ufl_domain._ufl_cargo
            self._geometry = self.mesh._geometry
            self._topology = self.mesh._topology
            
        self.createMeasures()
        self.gdim = self.mesh.geometry.dim
        self.tdim = self.mesh.topology.dim
        self.num_cells = len(self.mesh.topology.connectivity(self.tdim,0))

        # self.vols = np.array([df.Cell(self, i).volume() for i in range(self.num_cells())])
        self.dsN = {}
        self.dxR = {}

    def boundaries(self):
        return self.facet_tags
    
    def subdomains(self):
        return self.cell_tags
    
    def createMeasures(self):
         self.ds = ufl.Measure('ds', domain=self, subdomain_data=self.facet_tags)
         self.dx = ufl.Measure('dx', domain=self, subdomain_data=self.cell_tags)
         
