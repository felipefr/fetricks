#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:10:52 2026

@author: felipe
"""

import numpy as np
from mpi4py import MPI
import meshio 


class XDMFWriter(meshio.xdmf.TimeSeriesWriter):
    
    def __init__(self, mesh, filename):
        self.mesh = mesh
        self.X = self.mesh.geometry.x # points is reserved
        self.tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(self.tdim, 0) # "volume"-node connectivity
        self.cell_connect = self.mesh.topology.connectivity(self.tdim, 0).array.reshape(-1, self.tdim + 1) # cells is reserved

        # ---- Build meshio object ----
        cell_type_map = {
            1: "line",
            2: "triangle",
            3: "tetra"
        }

        self.cell_type = cell_type_map[self.tdim]

        self.cell_data_ = {} # cell_data is reserved    
        self.point_data_ = {} # cell_data is reserved
        self.fields_to_be_sync = [] # only if data is different than fenics object

        super().__init__(filename)        
    
    def register_field(self, u, export_dim=None):
        if(u.is_cellwise_constant()):
            self.cell_data_[u.name] = [u.x.array]
        else:
            dim = u.ufl_shape[0]     
            if( (export_dim is not None) and u.ufl_shape[0] < export_dim):
                n = int(len(u.x.array)/dim)
                data = np.empty((n, export_dim))
                data[:, u.ufl_shape[0]:] = 0.0
                self.point_data_[u.name] = data
                self.fields_to_be_sync.append((u.name, u.x.array, dim))
            else:
                self.point_data_[u.name] = u.x.array.reshape((-1, dim))
        
    def write_mesh(self):
        self.write_points_cells(self.X, [(self.cell_type, self.cell_connect)])
        
    def write_fields(self, step = 0):
        for (name, u_vals, dim) in self.fields_to_be_sync:
            self.point_data_[name][:, :dim] = u_vals.reshape((-1, dim))
        
        self.write_data(step, point_data = self.point_data_, cell_data = self.cell_data_)
        


class XDMFReader(meshio.xdmf.TimeSeriesReader):
    
    def __init__(self, filename):
        super().__init__(filename)
        self.X, self.cell_connect = self.read_points_cells()
        self.cell_connect = self.cell_connect[0].data.astype(np.int32)
        
    # def create_fenicsx_mesh(self, e):
    #     self.mesh = mesh.create_mesh(comm = MPI.COMM_WORLD,
    #                              cells = self.cell_connect,
    #                              x = self.X, 
    #                              e = e) # not working
        
    #     return self.mesh
    
    def read_field(self, u, k = 0):
        if(u.is_cellwise_constant()):
            u.x.array[:] = self.read_data(k)[2][u.name][0]
        else:
            u.x.array[:] = self.read_data(k)[1][u.name].flatten()
        

