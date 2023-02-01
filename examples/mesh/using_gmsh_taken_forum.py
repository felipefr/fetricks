#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:18:29 2023

@author: ffiguere
"""

import gmsh
import numpy as np

rectangle=[[0,0,0],[0,1,0],[3,1,0],[3,0,0]]
r_len=len(rectangle)
print(rectangle)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("mesh")

lc = 0.5

boundary_0=[0] #inflow
boundary_1=[2]  #outflow
boundary_2=[1,3] # wall

#add point
for i in range(r_len):
    gmsh.model.geo.addPoint(rectangle[i][0], rectangle[i][1], rectangle[i][2], lc, i)

#add line
gmsh.model.geo.addLine(0,1,0)
gmsh.model.geo.addLine(1,2,1)
gmsh.model.geo.addLine(2,3,2)
gmsh.model.geo.addLine(3,0,3)

#add physical group
b_0=gmsh.model.addPhysicalGroup(1,boundary_0,0)
b_1=gmsh.model.addPhysicalGroup(1,boundary_1,1)
b_2=gmsh.model.addPhysicalGroup(1,boundary_2,2)
gmsh.model.setPhysicalName(1,b_0,'inflow')
gmsh.model.setPhysicalName(1,b_1,'outflow')
gmsh.model.setPhysicalName(1,b_2,'wall')

#generate surface
gmsh.model.geo.addCurveLoop([0,1,2,3],1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.addPhysicalGroup(2,[1],1)

gmsh.model.geo.mesh.setAlgorithm(2,1,2)

#generate mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("mesh_test.msh")
gmsh.fltk.run()
gmsh.finalize()




import matplotlib.pyplot as plt
from dolfin import *
import numpy as np

import meshio
geometry = meshio.read("mesh_test.msh")
meshio.write("mesh_test.xdmf", meshio.Mesh(points=geometry.points, cells={"triangle": geometry.cells["triangle"]}))
meshio.write("mf_test.xdmf", meshio.Mesh(points=geometry.points, cells={"line": geometry.cells["line"]},
                                    cell_data={"line": {"name_to_read": geometry.cell_data["line"]["gmsh:physical"]}}))
#Load mesh and subdomains
mesh = Mesh()
with XDMFFile("mesh_test.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("mf_test.xdmf") as infile:
    infile.read(mvc, "name_to_read")
boundary_markers =MeshFunction("size_t",mesh, mvc)


#coefficient
Re=0.01
max_iteration=100

#Define function spaces
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = V * Q
W = FunctionSpace(mesh, TH)
P=W.sub(0).collapse()

#No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, boundary_markers, 0)

#Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
bc1 = DirichletBC(W.sub(0), inflow, boundary_markers, 1)

#Collect boundary conditions
bcs = [bc0, bc1]

#Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
u_n_minus_1=interpolate(f,P)
a = (inner(grad(u), grad(v))+dot(grad(u)*u_n_minus_1,v)- div(v)*p + q*div(u))*dx
L = inner(f, v)*dx
w = Function(W)

n=0
for n in range(max_iteration):
    #Update current iteration
    print("#################",n+1)
    #Compute solution
    solve(a == L, w, bcs)

    #Split the mixed solution using deepcopy
    #(needed for further computation on coefficient vector)
    (u, p) = w.split(True)

    #Compute error at vertices
    vertex_values_u_n_minus_1=u_n_minus_1.compute_vertex_values(mesh)
    vertex_values_u=u.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_u - vertex_values_u_n_minus_1))

    print(error_max)
    if error_max<1e-10:
        break
    #Update previous solution
    u_n_minus_1.assign(u)
    n+=1

#Split the mixed solution using a shallow copy
(u, p) = w.split()

#Save solution in VTK format
ufile_pvd = File("demo_stationary_ns/velocity.pvd")
ufile_pvd << u
pfile_pvd = File("demo_stationary_ns/pressure.pvd")
pfile_pvd << p

#Plot solution
plt.figure()
plot(u, title="velocity")

plt.figure()
plot(p, title="pressure")

#Display plots
plt.show()