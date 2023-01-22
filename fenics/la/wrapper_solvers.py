import dolfin as df
import numpy as np
from timeit import default_timer as timer

# Block Solver [K 0; 0 K] u = [F_1 ... F_N]
class BlockSolver:

    def __init__(self, subproblems):
        self.subproblems = subproblems
        self.n_subproblems = len(self.subproblems)
        
        self.F = [ df.PETScVector() for i in range(self.n_subproblems) ] 
        self.A = df.PETScMatrix()
        
        # supposing lhs and bcs are equal for all problems
        df.assemble(self.subproblems[0].a_ufl , tensor = self.A)
        [bc.apply(self.A) for bc in self.subproblems[0].bcs()] 
        
        self.solver = df.PETScLUSolver(self.A)

    def assembly_rhs(self):
        for i in range(self.n_subproblems): 
            df.assemble(self.subproblems[i].L_ufl, tensor = self.F[i])    
            [bc.apply(self.F[i]) for bc in self.subproblems[i].bcs()]
            
    def solve(self):   
        self.assembly_rhs()
        for i in range(self.n_subproblems): 
            self.solver.solve(self.subproblems[i].u_ufl.vector(), self.F[i])
            
class BlockSolverIndependent:

    def __init__(self, subproblems):
        self.subproblems = subproblems
        self.n_subproblems = len(self.subproblems)
        
        self.F = [ df.PETScVector() for i in range(self.n_subproblems) ] 
        
        # supposing lhs and bcs are equal for all problems
        self.solver = []
        self.A = []
        
        for i in range(self.n_subproblems):
            self.A.append(df.PETScMatrix())
            df.assemble(self.subproblems[i].a_ufl , tensor = self.A[-1])
            [bc.apply(self.A[-1]) for bc in self.subproblems[i].bcs()] 
            self.solver.append(df.PETScLUSolver(self.A[-1]))
            

    def assembly_rhs(self):
        for i in range(self.n_subproblems): 
            df.assemble(self.subproblems[i].L_ufl, tensor = self.F[i])    
            [bc.apply(self.F[i]) for bc in self.subproblems[i].bcs()]
            
    def solve(self):   
        self.assembly_rhs()
        for i in range(self.n_subproblems): 
            self.solver[i].solve(self.subproblems[i].u_ufl.vector(), self.F[i])

# Hand-coded implementation of Newton Raphson (Necessary in some cases)
def Newton(Jac, Res, bc, du, u, callbacks = [], Nitermax = 10, tol = 1e-8, u0_satisfybc = False):
    if(u0_satisfybc):
        for bc_i in bc: # non-homogeneous dirichlet applied only in the first itereation
            bc_i.homogenize()
        
    A, b = df.assemble_system(Jac, -Res, bc)
    
    # print("cond number", np.linalg.cond(A.array()))
    
    nRes = []
    nRes.append(b.norm("l2"))
    nRes[0] = nRes[0] if nRes[0]>0.0 else 1.0
    
    V = u.function_space()
    du.vector().set_local(np.zeros(V.dim()))
    # u.vector().set_local(np.zeros(V.dim()))
      
    niter = 0
    
    for bc_i in bc: # non-homogeneous dirichlet applied only in the first itereation
        bc_i.homogenize()
    
    while nRes[niter]/nRes[0] > tol and niter < Nitermax:
        df.solve(A, du.vector(), b)
        u.assign(u + du)
        for callback in callbacks:
            callback(u, du)
            
        A, b = df.assemble_system(Jac, -Res, bc)
        nRes.append(b.norm("l2"))
        print(" Residual:", nRes[niter+1])
        niter += 1
    
    return u, nRes

def NonlinearSolver(problem, bc, du, callbacks = [], Nitermax = 10, tol = 1e-8): 
    Jac = problem.J_ufl
    Res = problem.F_ufl
    u = problem.u_ufl
    
    return Newton(Jac, Res, bc, du, u, callbacks, Nitermax, tol)

# Automatic implementation of Newton Raphson (Necessary in some cases)
def Newton_automatic(Jac, Res, bc, du, u, callbacks = [], Nitermax = 10, tol = 1e-8):     
    problem = df.NonlinearVariationalProblem(Res, u, bc, Jac)
    microsolver = df.NonlinearVariationalSolver(problem)
    
    solver_parameters = {"nonlinear_solver": "newton",
                         "newton_solver": {"maximum_iterations": 20,
                                           "report": True,
                                           "error_on_nonconvergence": True}}

    microsolver.parameters.update(solver_parameters)

    microsolver.solve()
    return u


# def Newton(Jac, Res, bc, du, u, callbacks = None, Nitermax = 10, tol = 1e-8): 

#     A = df.PETScMatrix()
#     b = df.PETScVector()

#     df.assemble(Jac, tensor=A)
#     df.assemble(Res, tensor=b)

#     solver = df.LUSolver(A)
    
#     for bc_i in bc:
#         bc_i.apply(A,b)    

#     nRes0 = b.norm("l2")
#     nRes0 = nRes0 if nRes0>0.0 else 1.0
#     nRes = nRes0
    
#     V = u.function_space()
#     du.vector().set_local(np.zeros(len(du.vector().get_local())))
#     u.vector().set_local(np.zeros(len(du.vector().get_local())))
      
#     niter = 0
    
#     for bc_i in bc: # non-homogeneous dirichlet applied only in the first itereation
#         bc_i.homogenize()
    
#     while nRes/nRes0 > tol and niter < Nitermax:
#         solver.solve(du.vector(), b)
#         u.assign(u + du)
#         for callback in callbacks:
#             callback(u)
            
#         A, b = df.assemble_system(Jac, Res, bc)
#         nRes = b.norm("l2")
#         print(" Residual:", nRes)
#         niter += 1
    
#     return u

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

      
def local_project_given_sol(v, V, u=None, dxm = None):
    dv = df.TrialFunction(V)
    v_ = df.TestFunction(V)
    a_proj = df.inner(dv, v_)*dxm
    b_proj = df.inner(v, v_)*dxm
    solver = df.LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = df.Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return
    
    
# def local_project_given_sol(v, V, u, metadata = {}):
#     M = V.mesh()
#     dv = df.TrialFunction(V)
#     v_ = df.TestFunction(V)
#     dx = df.Measure('dx', M, metadata = metadata)
    
#     a_proj = df.inner(dv, v_)*dx
#     b_proj = df.inner(v, v_)*dx

#     solver = df.LocalSolver(a_proj)
#     solver.factorize()
    
#     b = df.assemble(b_proj)
    
#     solver.solve_local(u.vector(), b,  V.dofmap())

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
    def __init__(self, V, dx, sol = None):    
        self.V = V
        self.sol = sol if sol else df.Function(V)
        self.dx = dx
        
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        
        a_proj = df.inner(u, v)*self.dx
        self.b_proj = lambda w: df.inner(w, v)*self.dx
        
        self.rhs = df.PETScVector()
        
        self.solver = df.LocalSolver(a_proj)
        self.solver.factorize()
    
    
    def __call__(self, u):
        df.assemble(self.b_proj(u), tensor = self.rhs)
        self.solver.solve_local(self.sol.vector(), self.rhs,  self.V.dofmap())
        