# ============================================================================= 
# The class GmshIO provides basic IO basic functionalities for Gmsh
# using meshio and fenics. 
# 
# =============================================================================

import meshio
import os
import numpy as np

class GmshIO:
    def __init__(self, meshname = "default.xdmf", dim = 2):
        self.mesh = None
        self.dim = dim
        self.setNameMesh(meshname)
        
        self.gmsh_opt = '-format msh2 -{0}'.format(self.dim)
    
    def writeMSH(self, gmsh_opt = ''):
        meshGeoFile = self.radFileMesh.format('geo')
        meshMshFile = self.radFileMesh.format('msh')
    
        os.system('gmsh {0} {1} -o {2}'.format(self.gmsh_opt, meshGeoFile, meshMshFile))  # with del2d, noticed less distortions
        
        self.mesh = meshio.read(meshMshFile)
    
    def write(self, option = 'xdmf', optimize_storage = True):
        
        if(self.format == 'geo'):
            self.writeMSH()
            
        if(option == 'xdmf'):
            savefile = self.radFileMesh.format('xdmf')
            self.exportMeshHDF5(savefile, optimize_storage)
        else:
            meshXMLFile = self.radFileMesh.format('xml')
            meshMshFile = self.radFileMesh.format('msh')
            
            os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXMLFile))   
    
    def setNameMesh(self, meshname):
        self.radFileMesh,  self.format = meshname.split('.')
        self.radFileMesh += '.{0}'
        
    def __determine_geometry_types(self, mesh_msh):
        
        # trying to be generic for the type of cells, but for fenics 2019.1.0, working with quads it's almost impossible
        if(self.dim == 2):
            self.facet_type, self.cell_type = mesh_msh.cells_dict.keys()
            self.dummy_point = np.zeros((1,2))
            self.dummy_cell = np.arange(len(mesh_msh.cells_dict[self.cell_type][0]))
                
        elif(self.dim == 3):
            self.facet_type = "triangle"
            self.cell_type = "tetra"
            self.dummy_cell = np.array([[1,2,3,4]])
            self.dummy_point = np.zeros((1,3))

            # :
                # print("element type not recognised by fetricks")
            
    def exportMeshHDF5(self, meshFile = 'mesh.xdmf', optimize_storage = False):

        mesh_msh = meshio.read(self.radFileMesh.format('msh')) 
        
        self.__determine_geometry_types(mesh_msh)

        meshFileRad = meshFile[:-5]
        
        # working on mac, error with cell dictionary
        meshio.write(meshFile, meshio.Mesh(points=mesh_msh.points[:,:self.dim], 
                                           cells={self.cell_type: mesh_msh.cells_dict[self.cell_type]})) 
        
        self.__exportHDF5_faces(mesh_msh, meshFileRad, optimize = optimize_storage)
        self.__exportHDF5_regions(mesh_msh, meshFileRad, optimize = optimize_storage)

    def __exportHDF5_faces(self, mesh_msh, meshFileRad, optimize = False):
        mesh = self.__create_aux_mesh(mesh_msh, self.facet_type, 'faces', prune_z = (self.dim == 2) )   
        meshio.write("{0}_{1}.xdmf".format(meshFileRad,'faces'), mesh)

    def __exportHDF5_regions(self, mesh_msh, meshFileRad, optimize = False):        
        mesh = self.__create_aux_mesh(mesh_msh, self.cell_type, 'regions', prune_z = (self.dim == 2) )   
        meshio.write("{0}_{1}.xdmf".format(meshFileRad,'regions'), mesh)
        
    def __create_aux_mesh(self, mesh, cell_type, name_to_read, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={name_to_read:[cell_data]})
        return out_mesh
        
# self.mesh = pygmsh.generate_mesh(self, verbose=False, dim=2, prune_vertices=True, prune_z_0=True,
# remove_faces=False, extra_gmsh_arguments=gmsh_opt,  mesh_file_type='msh4') # it should be msh2 cause of tags