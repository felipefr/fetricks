# ============================================================================= 
# The class GmshIO provides basic IO basic functionalities for Gmsh
# using meshio and fenics. 
# 
# =============================================================================

import meshio
import os
import numpy as np

class gmshio:
    def __init__(self, meshname = "default.geo", dim = 2):
        self.mesh = None
        self.dim = dim
        self.set_name_mesh(meshname)
        
        self.gmsh_opt = '-{0}'.format(self.dim)
    
    def write_msh(self, gmsh_opt = None):
        meshGeoFile = self.radFileMesh.format('geo')
        meshMshFile = self.radFileMesh.format('msh')
    
        if(not gmsh_opt):
            gmsh_opt = self.gmsh_opt
            
        os.system('gmsh {0} {1} -o {2}'.format(gmsh_opt, meshGeoFile, meshMshFile))  # with del2d, noticed less distortions
        
        self.mesh = meshio.read(meshMshFile)
    
    def write_xdmf(self, optimize_storage = True):
        savefile = self.radFileMesh.format('xdmf')
        self.exportMeshHDF5(savefile, optimize_storage)
    
    def set_name_mesh(self, meshname):
        self.radFileMesh,  self.format = os.path.splitext(meshname)
        self.format = self.format[1:]
        self.radFileMesh += '.{0}'