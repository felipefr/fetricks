import dolfin as df
import numpy as np
from timeit import default_timer as timer

# Local projection is faster than the standard projection routine in DG spaces
def local_project(v,V):
    M = V.mesh()
    dv = df.TrialFunction(V)
    v_ = df.TestFunction(V)
    dx = df.Measure('dx', M)
    a_proj = df.inner(dv,v_)*dx 
    b_proj = df.inner(v,v_)*dx
    solver = df.LocalSolver(a_proj,b_proj) 
    solver.factorize()
    u = df.Function(V)
    solver.solve_local_rhs(u)
    return u
    

def local_project_metadata(v,V, metadata = {}):
    M = V.mesh()
    dv = df.TrialFunction(V)
    v_ = df.TestFunction(V)
    dx = df.Measure('dx', M, metadata = metadata)
    a_proj = df.inner(dv,v_)*dx 
    b_proj = df.inner(v,v_)*dx
    solver = df.LocalSolver(a_proj,b_proj) 
    solver.factorize()
    u = df.Function(V)
    solver.solve_local_rhs(u)
    return u

# PETSC krylov type solver with most common settings
def solver_iterative(a,b, bcs, Uh):
    uh = df.Function(Uh)
    
    # solver.solve()
    start = timer()
    A, F = df.assemble_system(a, b, bcs)
    end = timer()
    print("time assembling ", end - start)
    
    solver = df.PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters["relative_tolerance"] = 1e-5
    solver.parameters["absolute_tolerance"] = 1e-6
    # solver.parameters["nonzero_initial_guess"] = True
    solver.parameters["error_on_nonconvergence"] = False
    solver.parameters["maximum_iterations"] = 1000
    solver.parameters["monitor_convergence"] = True
    # solver.parameters["report"] = True
    # solver.parameters["preconditioner"]["ilu"]["fill_level"] = 1 # 
    solver.set_operator(A)
    solver.solve(uh.vector(), F)   

    return uh


# Direct solver (REMOVE?)
def solver_direct(a,b, bcs, Uh, method = "superlu" ):
    uh = df.Function(Uh)
    df.solve(a == b,uh, bcs = bcs, solver_parameters={"linear_solver": method})

    return uh
    
class LocalProjector:
    def __init__(self, V, dx):    
        self.dofmap = V.dofmap()
        
        dv = df.TrialFunction(V)
        v_ = df.TestFunction(V)
        
        a_proj = df.inner(dv, v_)*dx
        self.b_proj = lambda u: df.inner(u, v_)*dx
        
        self.solver = df.LocalSolver(a_proj)
        self.solver.factorize()
        
        self.sol = df.Function(V)
    
    def __call__(self, u, sol = None):
        b = df.assemble(self.b_proj(u))
        
        if sol is None:
            self.solver.solve_local(self.sol.vector(), b,  self.dofmap)
            return self.sol
        else:
            self.solver.solve_local(sol.vector(), b,  self.dofmap)
            return
    
# def local_project(v, V, dxm, u=None):
#     dv = df.TrialFunction(V)
#     v_ = df.TestFunction(V)
#     a_proj = df.inner(dv, v_)*dxm
#     b_proj = df.inner(v, v_)*dxm

#     solver = df.LocalSolver(a_proj)
#     solver.factorize()
    
#     b = df.assemble(b_proj)
    
#     if u is None:
#         u = df.Function(V)
#         solver.solve_local(u.vector(), b,  V.dofmap())
#         # solver.solve_local_rhs(u)
#         return u
#     else:
#         solver.solve_local(u.vector(), b,  V.dofmap())
#         return
