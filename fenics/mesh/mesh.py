import dolfin as df
from functools import reduce
import numpy as np

import fetricks.fenics.postprocessing.wrapper_io as iofe

class Mesh(df.Mesh):
    def __init__(self, mesh, comm = df.MPI.comm_world):
        super().__init__(comm)
    
        if(isinstance(mesh,str)):
            self.read_from_file(mesh, comm)
        else: # this should be only used for very simple meshes (rectangular ones)
            super().__init__(mesh) # call copy constructor, does not with comm
            self.boundaries, self.subdomains = self.label_boundaries() 
            
            
        self.createMeasures()
        self.vols = np.array([df.Cell(self, i).volume() for i in range(self.num_cells())])
        self.V = {}
        self.bcs = {}
        self.dsN = {}
        self.dxR = {}

    def read_from_file(self, meshfile, comm):
        if(meshfile[-3:] == 'xml'):
            df.File(meshfile) >> self            
            self.subdomains = df.MeshFunction("size_t", self, meshfile[:-4] + "_physical_region.xml")
            self.boundaries = df.MeshFunction("size_t", self, meshfile[:-4] + "_facet_region.xml")
            
        elif(meshfile[-4:] == 'xdmf'):
            self.subdomains, self.boundaries = iofe.readXDMF_with_markers(meshfile, self, comm)        
    
    def createMeasures(self):
        self.ds = df.Measure('ds', domain=self, subdomain_data=self.boundaries)
        self.dx = df.Measure('dx', domain=self, subdomain_data=self.subdomains)
    
    def createFiniteSpace(self,  spaceType = 'S', name = 'u', spaceFamily = 'CG', degree = 1):
        
        myFunctionSpace = df.TensorFunctionSpace if spaceType =='T' else (df.VectorFunctionSpace if spaceType == 'V' else df.FunctionSpace)
        
        self.V[name] = myFunctionSpace(self, spaceFamily, degree)
        
    def addDirichletBC(self, name = 'default', spaceName = 'u', g = df.Constant(0.0), markerLabel=0, sub = None):
        Vaux = self.V[spaceName] if type(sub)==type(None) else self.V[spaceName].sub(sub)
        self.bcs[name] = df.DirichletBC(Vaux, g , self.boundaries, markerLabel)
    
    def applyDirichletBCs(self,A,b = None):
        if(type(b) == type(None)):
            for bc in self.bcs.values():
                bc.apply(A)
        else:
            for bc in self.bcs.values():
                bc.apply(A,b)
                
    def nameNeumannBoundary(self, name, boundaryMarker):
        self.dsN[name] = reduce(lambda x,y: x+y, [self.ds(b) for b in boundaryMarker] )
        
    def nameRegion(self, name, regionMarker):
        self.dxR[name] = reduce(lambda x,y: x+y, [self.dx(r) for r in regionMarker] )
        
    
    
    # convention (0, left), (1, right), (2, bottom), (3, top), [if ndim = 3 (4, back), (5, front)], (unknown, 2*ndim )
    def label_boundaries(self):
        # x_min, x_max, y_min, y_max, ...
        min_max_X = np.array([np.min(self.coordinates(), axis = 0), np.max(self.coordinates(), axis = 0)]).T.flatten()
        ndim = int(len(min_max_X)/2)
        
        # codimension 1, label ndim*2 is reserved to faces not recognised
        boundary_labels = df.MeshFunction("size_t", self, dim= ndim - 1, value = ndim*2) 
        for i, x0 in enumerate(min_max_X):
            bnd = df.CompiledSubDomain('near(x[i], x0) && on_boundary', i = int(i/2),  x0 = x0)
            bnd.mark(boundary_labels, i)   
            
        # just a dummy one
        subdomains_labels = df.MeshFunction("size_t", self, dim= ndim, value = 0) # codimension 0
    
        return boundary_labels, subdomains_labels

    
