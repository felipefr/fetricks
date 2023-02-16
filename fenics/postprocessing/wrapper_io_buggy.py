#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:53:48 2023

@author: ffiguere


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import os
import dolfin as df
import h5py # unexpected behaviour without including it, although not explicitly used

def readXDMF_with_markers(meshFile, mesh, comm = df.MPI.comm_world):

    with df.XDMFFile(comm,meshFile) as infile:
        infile.read(mesh)
    
    mvc = df.MeshValueCollection("size_t", mesh, 1) # maybe is not generalised for 3d
    with df.XDMFFile(comm, "{0}_faces.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "faces")
                
    mf  = df.MeshFunction("size_t", mesh, mvc)
  
    mvc = df.MeshValueCollection("size_t", mesh, 2) # maybe is not generalised for 3d
    with df.XDMFFile(comm, "{0}_regions.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "regions")
    
    mt  = df.MeshFunction("size_t", mesh, mvc)
    
    return mt, mf
    
def exportXDMF_gen(filename, fields, k = 0, delete_old_file = True):
    if(delete_old_file):
        os.remove(filename)
        
    with df.XDMFFile(filename) as ofile: 
        ofile.parameters["flush_output"] = True
        ofile.parameters["functions_share_mesh"] = True
    
        
        if('vertex' in fields.keys()):
            for field in fields['vertex']:
                ofile.write(field, k) 
        
        if('cell' in fields.keys()):
            for field in fields['cell']:
                ofile.write(field, k) 

        if('cell_vector' in fields.keys()):
            for field in fields['cell_vector']:
                for field_i in field.split():
                    ofile.write(field_i, k) 
            
               
def exportXDMF_checkpoint_gen(filename, fields, k= 0, delete_old_file = True):
    
    if(delete_old_file):
        os.remove(filename)
        
    with df.XDMFFile(filename) as ofile: 
        ofile.parameters["flush_output"] = True
        ofile.parameters["functions_share_mesh"] = True
            
        if('vertex' in fields.keys()):
            for field in fields['vertex']:
                ofile.write_checkpoint(field, field.name(), k, df.XDMFFile.Encoding.HDF5, append = True)                
        
        if('cell' in fields.keys()):
            for field in fields['cell']:
                ofile.write_checkpoint(field, field.name(), k, df.XDMFFile.Encoding.HDF5, append = True) 


