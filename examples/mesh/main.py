import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import fetricks as ft 

import pygmsh
from timeit import default_timer as timer
from functools import partial 

df.parameters["form_compiler"]["representation"] = 'uflacs'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

from fetricks.fenics.mesh.mesh import Mesh 

from fetricks.fenics.mesh.wrapper_gmsh import Gmsh
import numpy as np

import meshio

" Adapted for the pygmsh 7.10"
class CookMembrane(Gmsh):
    def __init__(self, lcar = 1.0):
        super().__init__()    
              
        Lx = 48.0
        Hy = 44.0
        hy = 16.0
        
        hsplit = int(np.sqrt(Lx**2 + Hy**2)/lcar)
        vsplit = int(0.5*(Hy + hy)/lcar)
        
        p0 = self.add_point([0.0,0.0,0.0], lcar)
        p1 = self.add_point([Lx,Hy,0.0], lcar)
        p2 = self.add_point([Lx,Hy+hy,0.0], lcar)
        p3 = self.add_point([0.0,Hy,0.0], lcar)
        
        l0 = self.add_line(p0,p1)
        l1 = self.add_line(p1,p2)
        l2 = self.add_line(p2,p3)
        l3 = self.add_line(p3,p0)
        
        self.l = [l0,l1,l2,l3]
        a = self.add_curve_loop(self.l)
        self.s = self.add_surface(a)
        
        self.set_transfinite_curve(self.l[0], hsplit, "Progression", 1.0)
        self.set_transfinite_curve(self.l[2], hsplit, "Progression", 1.0)
        
        self.set_transfinite_curve(self.l[1], vsplit, "Progression", 1.0)
        self.set_transfinite_curve(self.l[3], vsplit, "Progression", 1.0)
        
        self.set_transfinite_surface(self.s, arrangement = 'alternate', corner_pts = [p0,p1,p2,p3])

        self.physicalNaming()
        
    def physicalNaming(self):
        self.add_physical(self.l[1], label = "1")
        self.add_physical(self.l[3], label = "2")
        self.add_physical(self.s, label = "0")



# mesh = CookMembrane(0.1)
# mesh.generate_mesh(dim = 2)


# with CookMembrane(1.0) as cm:
#     cm.generate_mesh(dim=2)


# OpenCascade in pygmsh seems not to support extraction of lines from a rectangle (... to use with physical labels).
# So, let's use the geo kernel:
    
# with pygmsh.geo.Geometry() as geom:
#     r1 = geom.add_rectangle(0., 5e-3, 0., 2.5e-3, z=0.)
#     geom.add_physical(r1.lines, label="1")
#     geom.add_physical(r1.surface, label="2")

#     mesh2 = geom.generate_mesh(dim=2)
    
# # We'll use gmsh format version 2.2 here, as there's a problem
# # with writing nodes in the format version 4.1 here, that I cannot figure out
# mesh.write("test.msh", file_format="gmsh22")

lcar = 1.0


    
def exportMeshXDMF(gmshMesh = 'mesh.msh', meshFile = 'mesh.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'}): #'meshTemp2.msh'    
    mesh = meshio.read(gmshMesh) if type(gmshMesh) == type('s') else gmshMesh
    
    meshFileRad = meshFile[:-5]
    
    # Export mesh without physical groups
    meshio.write(meshFile, meshio.Mesh(points=mesh.points[:,:2], cells=[('triangle', mesh.get_cells_type("triangle"))]) ) 
    
    # Export physical groups for boundaries
    meshFaces = meshio.Mesh(points= mesh.points[:,:2], cells=[('line', mesh.get_cells_type('line'))], 
                                                    cell_data={'Boundary': [mesh.cell_data_dict['gmsh:physical']['line']]} )
                                                     
    meshio.write("{0}_{1}.xdmf".format(meshFileRad,'faces'), meshFaces)
        
    # Export physical groups for regions 
    # mesh = meshio.Mesh(points=np.zeros((1,2)), cells={"triangle": np.array([[1,2,3]])}, cell_data={'triangle': {'regions': geometry.cell_data['triangle']["gmsh:physical"]}})
    
    # meshio.write("{0}_{1}.xdmf".format(meshFileRad,'regions'), mesh)
    
    # # hack to not repeat mesh information
    # f = h5py.File("{0}_{1}.h5".format(meshFileRad,'regions'),'r+')
    # del f['data1']
    # f['data1'] = h5py.ExternalLink(meshFileRad + ".h5", "data1")
    # f.close()
    
    # g = ET.parse("{0}_{1}.xdmf".format(meshFileRad,'regions'))
    # root = g.getroot()
    # root[0][0][2].attrib['NumberOfElements'] = root[0][0][3][0].attrib['Dimensions'] # left is topological in level, and right in attributes level
    # root[0][0][2][0].attrib['Dimensions'] = root[0][0][3][0].attrib['Dimensions'] + ' 3'
  
    # g.write("{0}_{1}.xdmf".format(meshFileRad,'regions'))


with pygmsh.geo.Geometry() as geom:
    Lx = 48.0
    Hy = 44.0
    hy = 16.0
    
    hsplit = int(np.sqrt(Lx**2 + Hy**2)/lcar)
    vsplit = int(0.5*(Hy + hy)/lcar)
    
    p0 = geom.add_point([0.0,0.0,0.0], lcar)
    p1 = geom.add_point([Lx,Hy,0.0], lcar)
    p2 = geom.add_point([Lx,Hy+hy,0.0], lcar)
    p3 = geom.add_point([0.0,Hy,0.0], lcar)
    
    l0 = geom.add_line(p0,p1)
    l1 = geom.add_line(p1,p2)
    l2 = geom.add_line(p2,p3)
    l3 = geom.add_line(p3,p0)
    
    l = [l0,l1,l2,l3]
    a = geom.add_curve_loop(l)
    s = geom.add_surface(a)

    geom.set_transfinite_curve(l[0], hsplit, "Progression", 1.0)
    geom.set_transfinite_curve(l[2], hsplit, "Progression", 1.0)
   
    geom.set_transfinite_curve(l[1], vsplit, "Progression", 1.0)
    geom.set_transfinite_curve(l[3], vsplit, "Progression", 1.0)
    
    geom.set_transfinite_surface(s, arrangement = 'alternate', corner_pts = [p0,p1,p2,p3])
    
    geom.add_physical(l[1], label = "Left")
    geom.add_physical(l[3], label = "Right")
    geom.add_physical(s, label = "Volume")
    
    mesh = geom.generate_mesh()
    
    mesh.write("test.msh", file_format="gmsh22")
    

outfile_mesh = "test.xdmf"
mesh = meshio.read('test.msh')

exportMeshXDMF(mesh, outfile_mesh)


# outfile_boundary = "test_boundaries.xdmf"

# # delete third (obj=2) column (axis=1), this strips the z-component
# outpoints = np.delete(arr=mesh.points, obj=2, axis=1)

# # create (two dimensional!) triangle mesh file
# outmsh = meshio.Mesh(points=outpoints,
#                       cells=[('triangle', mesh.get_cells_type("triangle"))],
#                       cell_data={'Subdomain': [mesh.cell_data_dict['gmsh:physical']['triangle']]},
#                       field_data=mesh.field_data)

# # write mesh to file
# meshio.write(outfile_mesh, outmsh)
# # create (two dimensional!) boundary data file
# outboundary = meshio.Mesh(points=outpoints,
#                            cells=[('line', mesh.get_cells_type('line') )],
#                            cell_data={'Boundary': [mesh.cell_data_dict['gmsh:physical']['line']]},
#                            field_data=mesh.field_data)
# # write boundary data to file
# meshio.write(filename=outfile_boundary, mesh=outboundary)


# elastic parameters

E = 100.0
nu = 0.3
alpha = 200.0
ty = 5.0

model = ft.hyperelasticModel({'E': E, 'nu': nu, 'alpha': alpha})

mesh = Mesh(outfile_mesh)

start = timer()

clampedBndFlag = 2 
LoadBndFlag = 1 
traction = df.Constant((0.0,ty ))
    
deg_u = 1
deg_stress = 0
V = df.VectorFunctionSpace(mesh, "CG", deg_u)
We = df.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
W = df.FunctionSpace(mesh, We)
W0e = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = df.FunctionSpace(mesh, W0e)

bcL = df.DirichletBC(V, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
bc = [bcL]

def F_ext(v):
    return df.inner(traction, v)*mesh.ds(LoadBndFlag)


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = df.dx(metadata=metadata)

model.createInternalVariables(W, W0, dxm)
u = df.Function(V, name="Total displacement")
du = df.Function(V, name="Iteration correction")
v = df.TestFunction(V)
u_ = df.TrialFunction(V)

a_Newton = df.inner(ft.tensor2mandel(ft.symgrad(u_)), model.tangent(ft.tensor2mandel(ft.symgrad(v))) )*dxm
res = -df.inner(ft.tensor2mandel(ft.symgrad(v)), model.sig )*dxm + F_ext(v)

file_results = df.XDMFFile("cook.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


callbacks = [lambda w: model.update_alpha(ft.tensor2mandel(ft.symgrad(w))) ]

ft.Newton(a_Newton, res, bc, du, u, callbacks , Nitermax = 10, tol = 1e-8)

## Solve here Newton Raphson

file_results.write(u, 0.0)

end = timer()
print(end - start)