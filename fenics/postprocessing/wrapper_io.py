from __future__ import print_function
import numpy as np
from fenics import *
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import sys, os
import copy

from functools import reduce

from timeit import default_timer as timer
import meshio
import h5py
import xml.etree.ElementTree as ET

def readXDMF_with_markers(meshFile, mesh, comm = MPI.comm_world):

    with XDMFFile(comm,meshFile) as infile:
        infile.read(mesh)
    
    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(comm, "{0}_faces.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "faces")
                
    mf  = MeshFunction("size_t", mesh, mvc)
  
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(comm, "{0}_regions.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "regions")
    
    mt  = MeshFunction("size_t", mesh, mvc)
    
    # return mt, mf
    return mt, mf

def exportMeshHDF5_fromGMSH(gmshMesh = 'mesh.msh', meshFile = 'mesh.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'}): #'meshTemp2.msh'
    # Todo: Export just mesh with meshio then update .h5 file with domains and boundary physical markers. Need of just one xdmf file, and also h5 ==> look below
    # import dolfin as df

    # mesh = df.Mesh(xml_mesh_name)
    # mesh_file = df.HDF5File(df.mpi_comm_world(), h5_file_name, 'w')
    # mesh_file.write(mesh, '/mesh')
    
    # # maybe you have defined a mesh-function (boundaries, domains ec.)
    # # in the xml_mesh aswell, in this case use the following two lines
    
    # domains = df.MeshFunction("size_t", mesh, 3, mesh.domains())
    # mesh_file.write(domains, "/domains")
    # read in parallel:
    
    # mesh = df.Mesh()
    # hdf5 = df.HDF5File(df.mpi_comm_world(), h5_file_name, 'r')
    # hdf5.read(mesh, '/mesh', False)
    
    # # in case mesh-functions are available ...
    
    # domains = df.CellFunction("size_t", mesh)
    # hdf5.read(domains, "/domains")
    
    geometry = meshio.read(gmshMesh) if type(gmshMesh) == type('s') else gmshMesh
    
    meshFileRad = meshFile[:-5]
    
    meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:2], cells={"triangle": geometry.cells["triangle"]})) # working on mac, error with cell dictionary
    # meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:2], cells={"triangle": geometry.cells}))
        
    mesh = meshio.Mesh(points=np.zeros((1,2)), cells={'line': geometry.cells['line']},
                                                                              cell_data={'line': {'faces': geometry.cell_data['line']["gmsh:physical"]}})
    
    meshio.write("{0}_{1}.xdmf".format(meshFileRad,'faces'), mesh)
    
    # f = h5py.File("{0}_{1}.h5".format(meshFileRad,'faces'),'r+')
    # del f['data0']
    # f['data0'] = h5py.ExternalLink(meshFileRad + ".h5", "data0")
    # f.close()
    
    mesh = meshio.Mesh(points=np.zeros((1,2)), cells={"triangle": np.array([[1,2,3]])}, cell_data={'triangle': {'regions': geometry.cell_data['triangle']["gmsh:physical"]}})
    
    meshio.write("{0}_{1}.xdmf".format(meshFileRad,'regions'), mesh)
    
    # hack to not repeat mesh information
    f = h5py.File("{0}_{1}.h5".format(meshFileRad,'regions'),'r+')
    del f['data1']
    f['data1'] = h5py.ExternalLink(meshFileRad + ".h5", "data1")
    f.close()
    
    g = ET.parse("{0}_{1}.xdmf".format(meshFileRad,'regions'))
    root = g.getroot()
    root[0][0][2].attrib['NumberOfElements'] = root[0][0][3][0].attrib['Dimensions'] # left is topological in level, and right in attributes level
    root[0][0][2][0].attrib['Dimensions'] = root[0][0][3][0].attrib['Dimensions'] + ' 3'
  
    g.write("{0}_{1}.xdmf".format(meshFileRad,'regions'))
    

def exportXDMF_gen(filename, fields):
    with XDMFFile(filename) as ofile: 
        ofile.parameters["flush_output"] = True
        ofile.parameters["functions_share_mesh"] = True

            
        if('vertex' in fields.keys()):
            for field in fields['vertex']:
                ofile.write(field, 0.) 
        

        if('cell' in fields.keys()):
            for field in fields['cell']:
                for field_i in field.split():
                    ofile.write(field_i, 0.) 
                    
def exportXDMF_checkpoint_gen(filename, fields):
    with XDMFFile(filename) as ofile: 
        ofile.parameters["flush_output"] = True
        ofile.parameters["functions_share_mesh"] = True
            
        count = 0
        if('vertex' in fields.keys()):
            for field in fields['vertex']:
                ofile.write_checkpoint(field, field.name(), count, append = True)
                # count = count + 1
        

        if('cell' in fields.keys()):
            for field in fields['cell']:
                for field_i in field.split():
                    ofile.write_checkpoint(field_i, field_i.name(), count, append = True) 
                    # count = count + 1