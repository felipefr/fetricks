# Example truss

import os, sys
os.environ['Hufl5_DISABLE_VERSION_CHECK']='2'
sys.path.append("/home/felipe/sources/fetricksx")

import ufl 
from dolfinx import fem, io, plot
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import fetricksx as ft
import gmsh
from functools import partial
import dolfinx.fem.petsc

def get_mesh_truss(X, cells, param):   
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
    geom = gmsh.model.geo
    
    points = []
    for i, v in enumerate(X):
        if(X.shape[1]==2):
            points.append(geom.add_point(v[0], v[1], 0.0))
        elif(X.shape[1]==3):
            points.append(geom.add_point(v[0], v[1], v[2]))
        
    lines = []
    for i, c in enumerate(cells):
        lines.append(geom.add_line(c[0] + 1, c[1] + 1))
    
    
    geom.synchronize()
    
    for l in lines:
         gmsh.model.mesh.set_transfinite_curve(l, 2)
    
    
    gmsh.model.add_physical_group(1, lines, 0)
    
    for phy in param["physical_groups"]:
        gmsh.model.add_physical_group(phy[0], phy[1], phy[2])
        

    
    gmsh.model.mesh.generate(dim=1)
    gmsh.write("truss.msh")
    
    domain, markers, facets = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    
    gmsh.finalize()
    
    return domain, markers, facets



def get_tangent_truss(domain):
    gdim = 2
    # J = ufl.Jacobian(domain)
    J = ufl.Jacobian(domain)[:, 0]
    sh0 = fem.functionspace(domain, ("DG", 0, (gdim,)))
    t_ufl = J/ufl.sqrt(ufl.inner(J,J))
    t = fem.Function(sh0, name="Tangent_vector")
    t.interpolate(fem.Expression(t_ufl, sh0.element.interpolation_points()))
    
    return t

def grad_truss(u, t):
    return ufl.dot(ufl.grad(ufl.dot(u,t)), t) # inner( outer(t,t) , grad(u)) 

def solve_truss(param, domain):
    # domain, markers, facets = get_mesh_truss(param['X'], param['cells'])
        
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
        
    Area = param['Area']
    sigma_law = param['sigma_law']
    
    t = get_tangent_truss(domain)
    
    grad = partial(grad_truss, t = t)
    
    Uh = fem.functionspace(domain, ("CG", 1, (gdim,)))
    
    bcs_D = [ fem.dirichletbc(bc[2], np.array([bc[0]*gdim + bc[1]]).astype('int32'), Uh.sub(bc[1])) for bc in param['dirichlet'] ]
    
    load_vec = fem.Function(Uh)
    fdim = tdim - 1
    for bc in param['neumann']: 
        load_vec.x.array[bc[0]*gdim + bc[1]] = bc[2]
     
    # # # Define variational problem
    uh = ufl.TrialFunction(Uh) 
    vh = ufl.TestFunction(Uh)
    
    dx = ufl.Measure('dx', domain = domain)
    
    a_form = fem.form(ufl.inner(Area*sigma_law(grad(uh)),  grad(vh))*dx)
    
    F0 = fem.Constant(domain, np.zeros((gdim,)))
    L_form = fem.form(ufl.dot(F0, vh) * dx)

    A = fem.petsc.assemble_matrix(a_form, bcs=bcs_D)
    A.assemble()
    b = fem.petsc.create_vector(L_form)
    
    b.array[:] = load_vec.x.array[:]
    # -

    # ### Solving the problem
    #
    # We finally setup the linear solver and solve the problem.

    # +
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    u = fem.Function(Uh, name="Displacement")

    solver.solve(b, u.vector)
    u.x.scatter_forward()

    return u


def posproc_truss(param, uh, domain):
#    domain, markers, facets = get_mesh_truss(param['X'], param['cells'])

    t = get_tangent_truss(domain)
    grad = partial(grad_truss, t = t)
    sigma_law = param['sigma_law']
    
    strain_ufl = grad(uh)
    stress_ufl = sigma_law(strain_ufl)
    
    
    sh0 = fem.functionspace(domain, ("DG", 0, ()))
    stress_exp = fem.Expression(stress_ufl, sh0.element.interpolation_points())
    stress_h = fem.Function(sh0, name="stress")
    stress_h.interpolate(stress_exp)

    strain_exp = fem.Expression(strain_ufl, sh0.element.interpolation_points())
    strain_h = fem.Function(sh0, name="strain")
    strain_h.interpolate(strain_exp)

    return {'strain': strain_h, 'stress': stress_h}

if __name__ == '__main__':
    
    L = 3.6 
    X = np.array([[0., 0.],
                    [L, 0.],
                    [2*L, 0.],
                    [0., L],
                    [L, L],
                    [2*L, L]])

    cells = np.array([  [0, 1],
                        [1, 2],
                        [3, 4],
                        [4, 5],
                        [2, 5],
                        [1, 4],
                        [1, 5],
                        [2, 4],
                        [0, 4],
                        [1, 3]], dtype = int)
    
    
    fy = -1.0
    # p1 = ufl.Point((0.0, 0.0)) # node 0
    # p2 = ufl.Point((0, L)) # node 3           
    # p3 = ufl.Point((L, 0.0)) # node 1
    # p4 = ufl.Point((2*L, 0.0)) # node 2           
    E = 100.0
    

# for dirichlet and neumann: tuple of (node (int or ufl.Point), direction, value)
    param = {
    'physical_groups': [(0,[0],1), (0,[3],2), (0,[1],3), (0,[2],4) ],
    'dirichlet': [(0, 0, 0. ), (0, 1, 0. ),
                  (3, 0, 0. ), (3, 1, 0. )],
    'neumann': [(1, 1 , fy), (2, 1 , fy)], 
    'X': X,
    'cells': cells
    }
    
    domain, markers, facets = get_mesh_truss(X, cells, param)
    param['sigma_law'] =  lambda e: fem.Constant(domain, E)*e
    param['Area'] = fem.Constant(domain, 1.0)

    uh = solve_truss(param, domain)
    zh = posproc_truss(param, uh, domain)
    
    with io.XDMFFile(MPI.COMM_WORLD, "truss.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
        xdmf.write_function(zh["stress"])
        # xdmf.write_function(zh["strain"])
            
    # uh.rename('u', '')
    
    # ft.exportXDMF_gen("truss_vtk_2.xdmf", fields={'vertex': [uh], 'cell': [zh['strain'], zh['stress']]})
    


