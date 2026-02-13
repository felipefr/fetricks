# ============================================================================= 
# The class Gmsh is wrapper for pygmsh Geometry. It provides basic functionalities
# to interect with meshio and fenics. 
# 
# =============================================================================

import numpy as np
import meshio
import pygmsh
import os

# Instructions : generation from gmsh
# gmsh -format 'msh22' -3 quarterCilinder.geo -o quarterCilinder.msh
# dolfin-convert quarterCilinder.msh quarterCilinder.xml

class GmshIO(pygmsh.built_in.Geometry):
    def __init__(self, meshname = "default.xdmf", dim = 2):
        super().__init__()   
        self.mesh = None
        self.dim = dim
        self.setNameMesh(meshname)
        self.gmsh_opt = '-format msh2 -{0}'.format(self.dim)
        
        if(self.format == "geo"):
            f = open(meshname)
            self.add_raw_code(f.read())
            f.close()
        
    # write .geo file necessary to produce msh files using gmsh
    def writeGeo(self):
        savefile = self.radFileMesh.format('geo')
        f = open(savefile,'w')
        f.write(self.get_code())
        f.close()   
    
    def writeMSH(self, gmsh_opt = ''):
        meshGeoFile = self.radFileMesh.format('geo')
        meshMshFile = self.radFileMesh.format('msh')
    
        os.system('gmsh {0} {1} -o {2}'.format(self.gmsh_opt, meshGeoFile, meshMshFile))  # with del2d, noticed less distortions
        
        self.mesh = meshio.read(meshMshFile)
        
        
    def write(self, option = 'xdmf', optimize_storage = True):
        
        if(self.format == 'geo'):
            self.writeMSH()
            
        elif(type(self.mesh) == type(None)):
            self.generate()
            
        if(option == 'xdmf'):
            savefile = self.radFileMesh.format('xdmf')
            self.exportMeshHDF5(savefile, optimize_storage)
        else:
            meshXMLFile = self.radFileMesh.format('xml')
            meshMshFile = self.radFileMesh.format('msh')
            
            os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXMLFile))   
  
    def generate(self):
        self.mesh = pygmsh.generate_mesh(self, verbose = False,
                    extra_gmsh_arguments = self.gmsh_opt.split(), dim = self.dim, 
                    mesh_file_type = 'msh2') # it should be msh2 cause of tags    
                                          

    def setNameMesh(self, meshname):
        self.radFileMesh,  self.format = meshname.split('.')
        self.radFileMesh += '.{0}'
        
    def __determine_geometry_types(self):
        if(self.dim == 2):
            self.facet_type = "line"
            self.cell_type = "triangle"
            self.dummy_cell = np.array([[1,2,3]])
            self.dummy_point = np.zeros((1,2))
        elif(self.dim == 3):
            self.facet_type = "triangle"
            self.cell_type = "tetra"
            self.dummy_cell = np.array([[1,2,3,4]])
            self.dummy_point = np.zeros((1,3))
            
    def exportMeshHDF5(self, meshFile = 'mesh.xdmf', optimize_storage = False):

        self.__determine_geometry_types()

        geometry = meshio.read(self.mesh) if type(self.mesh) == type('s') else self.mesh
        
        meshFileRad = meshFile[:-5]
        
        # working on mac, error with cell dictionary
        meshio.write(meshFile, meshio.Mesh(points=geometry.points[:,:self.dim], cells={self.cell_type: geometry.cells[self.cell_type]})) 
    
        self.__exportHDF5_faces(geometry, meshFileRad, optimize = optimize_storage)
        self.__exportHDF5_regions(geometry, meshFileRad, optimize = optimize_storage)

        if(optimize_storage):
            self.__hack_exportHDF5_regions(meshFileRad)

    def __exportHDF5_faces(self, geometry, meshFileRad, optimize = False):
        mesh = meshio.Mesh(points= self.dummy_point if optimize else geometry.points[:,:self.dim], 
                           cells={self.facet_type: geometry.cells[self.facet_type]},
                           cell_data={self.facet_type: {'faces': geometry.cell_data[self.facet_type]["gmsh:physical"]}})
        
        meshio.write("{0}_{1}.xdmf".format(meshFileRad,'faces'), mesh)

    def __exportHDF5_regions(self, geometry, meshFileRad, optimize = False):        
        mesh = meshio.Mesh(points= self.dummy_point if optimize else geometry.points[:,:self.dim], 
                           cells={self.cell_type: self.dummy_cell if optimize else geometry.cells[self.cell_type] }, 
                           cell_data={self.cell_type: {'regions': geometry.cell_data[self.cell_type]["gmsh:physical"]}})
        
        meshio.write("{0}_{1}.xdmf".format(meshFileRad,'regions'), mesh)

    def __hack_exportHDF5_regions(self, meshFileRad):  
        import h5py
        import xml.etree.ElementTree as ET
    
        # hack to not repeat mesh information
        f = h5py.File("{0}_{1}.h5".format(meshFileRad,'regions'),'r+')
        del f['data1']
        f['data1'] = h5py.ExternalLink(meshFileRad + ".h5", "data1")
        f.close()
        
        g = ET.parse("{0}_{1}.xdmf".format(meshFileRad,'regions'))
        root = g.getroot()
        root[0][0][2].attrib['NumberOfElements'] = root[0][0][3][0].attrib['Dimensions'] # left is topological in level, and right in attributes level
        root[0][0][2][0].attrib['Dimensions'] = root[0][0][3][0].attrib['Dimensions'] + ' ' + str(len(self.dummy_cell[0]))
      
        g.write("{0}_{1}.xdmf".format(meshFileRad,'regions'))
        
        
