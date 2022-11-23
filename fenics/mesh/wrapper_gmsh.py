# ============================================================================= 
# The class Gmsh is wrapper for pygmsh Geometry. It provides basic functionalities
# to interect with meshio and fenics. 
# 
# =============================================================================

import numpy as np
import meshio
import h5py
import xml.etree.ElementTree as ET
import pygmsh
import os
import dolfin as df
from functools import reduce
from fetricks.fenics.mesh.mesh import Mesh


# Instructions : generation from gmsh
# gmsh -format 'msh22' -3 quarterCilinder.geo -o quarterCilinder.msh
# dolfin-convert quarterCilinder.msh quarterCilinder.xml

class Gmsh(pygmsh.built_in.Geometry):
    def __init__(self, meshname = "default.xdmf", dim = 2):
        super().__init__()   
        self.mesh = None
        self.dim = dim
        self.setNameMesh(meshname)
        self.gmsh_opt = '-format msh2 -{0} -smooth 2 -anisoMax 1000.0'.format(self.dim)
        
        
    # write .geo file necessary to produce msh files using gmsh
    def writeGeo(self):
        savefile = self.radFileMesh.format('geo')
        f = open(savefile,'w')
        f.write(self.get_code())
        f.close()

    # write .xml files (standard for fenics): 
    def writeXML(self):
        meshXMLFile = self.radFileMesh.format('xml')
        meshMshFile = self.radFileMesh.format('msh')
        self.writeMSH()
        
        os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXMLFile))    
    
    def writeMSH(self, gmsh_opt = ''):
        self.writeGeo()
        meshGeoFile = self.radFileMesh.format('geo')
        meshMshFile = self.radFileMesh.format('msh')
    
        os.system('gmsh {0} {1} -o {2}'.format(self.gmsh_opt, meshGeoFile, meshMshFile))  # with del2d, noticed less distortions
        
        self.mesh = meshio.read(meshMshFile)
        
    def write(self, opt = 'meshio'):
        if(type(self.mesh) == type(None)):
            self.generate()
        if(opt == 'meshio'):
            savefile = self.radFileMesh.format('msh')
            meshio.write(savefile, self.mesh)
        elif(opt == 'fenics'):
            savefile = self.radFileMesh.format('xdmf')
            self.exportMeshHDF5(savefile)
            
    def generate(self):
        self.mesh = pygmsh.generate_mesh(self, verbose = False,
                    extra_gmsh_arguments = self.gmsh_opt.split(), dim = self.dim, 
                    mesh_file_type = 'msh2') # it should be msh2 cause of tags    
                                          

    def setNameMesh(self, meshname):
        self.radFileMesh,  self.format = meshname.split('.')
        self.radFileMesh += '.{0}'
        
        
    def exportMeshHDF5(self, meshFile = 'mesh.xdmf'):
    
        if(self.dim == 2):
            facet_type = "line"
            cell_type = "triangle"
            dummy_cell = np.array([[1,2,3]])
            dummy_point = np.zeros((1,2))
        elif(self.dim == 3):
            facet_type = "triangle"
            cell_type = "tetra"
            dummy_cell = np.array([[1,2,3,4]])
            dummy_point = np.zeros((1,3))
            
        
        geometry = meshio.read(self.mesh) if type(self.mesh) == type('s') else self.mesh
        
        meshFileRad = meshFile[:-5]
        
        # working on mac, error with cell dictionary
        meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:self.dim], cells={cell_type: geometry.cells[cell_type]})) 
    
        mesh = meshio.Mesh(points=dummy_point, cells={facet_type: geometry.cells[facet_type]},
                                                   cell_data={facet_type: {'faces': geometry.cell_data[facet_type]["gmsh:physical"]}})
        
        meshio.write("{0}_{1}.xdmf".format(meshFileRad,'faces'), mesh)
            
        mesh = meshio.Mesh(points=dummy_point, cells={cell_type: dummy_cell}, 
                                                   cell_data={cell_type: {'regions': geometry.cell_data[cell_type]["gmsh:physical"]}})
        
        meshio.write("{0}_{1}.xdmf".format(meshFileRad,'regions'), mesh)
        
        # hack to not repeat mesh information
        f = h5py.File("{0}_{1}.h5".format(meshFileRad,'regions'),'r+')
        del f['data1']
        f['data1'] = h5py.ExternalLink(meshFileRad + ".h5", "data1")
        f.close()
        
        g = ET.parse("{0}_{1}.xdmf".format(meshFileRad,'regions'))
        root = g.getroot()
        root[0][0][2].attrib['NumberOfElements'] = root[0][0][3][0].attrib['Dimensions'] # left is topological in level, and right in attributes level
        root[0][0][2][0].attrib['Dimensions'] = root[0][0][3][0].attrib['Dimensions'] + ' ' + str(len(dummy_cell[0]))
      
        g.write("{0}_{1}.xdmf".format(meshFileRad,'regions'))
