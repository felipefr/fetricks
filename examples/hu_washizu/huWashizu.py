import numpy as np
import dolfin as df
import fetricks as ft
import fetricks.mechanics.elasticity_conversions as mech
import fetricks.fenics.postprocessing.wrapper_io as iofe
from fetricks.fenics.la.wrapper_solvers import local_project


# Solver parameters
# solver_parameters = {"nonlinear_solver": "snes",
#                           "snes_solver": {"linear_solver": "superlu",
#                                           "maximum_iterations": 20,
#                                           "report": True,
#                                           "error_on_nonconvergence": True}}

#

solver_parameters = {"nonlinear_solver": "newton",
                     "newton_solver": {"maximum_iterations": 20,
                                       "report": True,
                                       "error_on_nonconvergence": True}}
                                       
df.parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}

folder = './meshes/'
meshfile = folder + 'mesh_40.xdmf'

mesh = ft.Mesh(meshfile)

clampedBndFlag = 2 
LoadBndFlag = 1 

ty = 2.0
traction = df.Constant((0.0,ty ))

metric = {'YOUNG_MODULUS': 120.0,
          'POISSON_RATIO': 0.3}

lamb, mu = mech.youngPoisson2lame(metric['POISSON_RATIO'], metric['YOUNG_MODULUS']) 
alpha = 200.0


# in intrisic notation
def psi(e): 
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    psi = 0.5*lamb*(tr_e**2 + 0.5*alpha*tr_e**4) + mu*(e2 + 0.5*alpha*e2**2)
     
    return psi


def sig(e): 
    e = df.variable(e)
    sig = df.diff(psi(e),e)
    
    return sig


l = lambda v: df.inner(traction, v)*mesh.ds(LoadBndFlag)   
a = lambda e, t: df.inner(sig(e), t)*mesh.dx # only used if you write residuals directly
c = lambda v, t: df.inner(ft.symgrad(v), t)*mesh.dx
b = lambda e, t: df.inner(e, t)*mesh.dx    
Psi = lambda e: psi(e)*mesh.dx
    

# Classical Three-field Hu-Washizu formulation (u, e, s) Ptot = 0.5*(C*e)*e - e*s + symgrad(u)*s - l(u)
# Replace by Psi the elastic 0.5*(C*e)*e if nonlinear
def HuWashizu():
    # Function spaces

    Ue = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    Se = df.TensorElement("DG", mesh.ufl_cell(), 0, symmetry = True)
        
    We = df.MixedElement(Ue, Se, Se)
    Wh = df.FunctionSpace(mesh, We)

    # # Boundary conditions
    bcL = df.DirichletBC(Wh.sub(0), df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
    bc = [bcL]
    
    # Test and trial functions: monolithic
    w = df.Function(Wh)
    (uh, eh, sh) = df.split(w)
        
    Pi = Psi(eh) - b(eh,sh) + c(uh,sh) - l(uh)
    F = df.derivative(Pi, w) 
    J = df.derivative(F, w)

    # # Solve
    problem = df.NonlinearVariationalProblem(F, w, bc, J)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters.update(solver_parameters)
    solver.solve()
    
    return  w.split()


# Primal formutation of elasticity ( in voigt)
def Primal():
    # Function spaces

    Ue = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    
    Uh = df.FunctionSpace(mesh, Ue)
    
    bcL = df.DirichletBC(Uh, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
    bc = [bcL]
    
    uh = df.Function(Uh)
    
    Pi = Psi(ft.symgrad(uh)) - l(uh)
    F = df.derivative(Pi, uh) 
    J = df.derivative(F, uh)
    
    problem = df.NonlinearVariationalProblem(F, uh, bcs = bc, J = J)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters.update(solver_parameters)
    solver.solve()
    
    Se = df.TensorElement("DG", mesh.ufl_cell(), 0, symmetry = True)
    Sh = df.FunctionSpace(mesh, Se)
    eh = local_project(ft.symgrad(uh), Sh)
    sh = local_project(sig(ft.symgrad(uh)), Sh)
    
    return (uh, eh, sh)


(u, e, s) = HuWashizu()
(u_, e_, s_) = Primal()

error_u = np.sqrt(df.assemble(df.inner(u - u_, u - u_)*mesh.dx))
error_eps = np.sqrt(df.assemble(df.inner(e - e_, e - e_)*mesh.dx))
error_sig = np.sqrt(df.assemble(df.inner(s - s_, s - s_)*mesh.dx))

print("error u:" , error_u )
print("error eps:" , error_eps) 
print("error sig:" , error_sig)

assert np.allclose(error_u, 0.0)
assert np.allclose(error_eps, 0.0)
assert np.allclose(error_sig, 0.0)
      
s.rename('sigma', '')
e.rename('eps', '')
u.rename('u', '')
iofe.exportXDMF_gen("cook_vtk.xdmf", fields={'vertex': [u] , 'cell_vector': [e, s]})
