
"""
This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import dolfin as df
import numpy as np
from timeit import default_timer as timer


class CustomNonlinearProblem(df.NonlinearProblem):
    def __init__(self, L, u, bcs, a, callbacks = []):
        df.NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs
        self.u = u
        
    def F(self, b, x = None): 
        df.assemble(self.L, tensor=b)
        [bc.apply(b) for bc in self.bcs] 
        
    def J(self, A, x = None): 
        df.assemble(self.a, tensor=A)
        [bc.apply(A) for bc in self.bcs]


# problem should be a NonlinearVariationalProblem
class CustomNonlinearSolver:

    def __init__(self, problem, callbacks = [], u0_satisfybc = False): 
        
        self.problem = problem
        self.callbacks = callbacks
        
        self.du = df.Function(self.problem.u.function_space())
        self.du.vector().set_local(np.zeros(self.du.vector().size()))
        
        self.rhs = df.PETScVector() 
        self.lhs = df.PETScMatrix()
        
    def solve(self, Nitermax = 10, tol = 1e-8, report = True):
        
        self.call_callbacks()
        nRes = []
        nRes.append(self.assemble())
        
        nRes[0] = nRes[0] if nRes[0]>0.0 else 1.0
        niter = 0
        
        while nRes[niter]/nRes[0] > tol and niter < Nitermax:
            nRes.append(self.newton_raphson_iteration())
            niter+=1
            
            if(report):
                print(" Residual:", nRes[-1]/nRes[0])
            
            
        return nRes

    def newton_raphson_iteration(self):
        df.solve(self.lhs, self.du.vector(), self.rhs)
        self.problem.u.assign(self.problem.u + self.du)    
        self.call_callbacks()
        return self.assemble()
        
    def call_callbacks(self):
        [foo(self.problem.u, self.du) for foo in self.callbacks]
    
    def assemble(self):
        self.problem.F(self.rhs)
        self.problem.J(self.lhs)
        
        return self.rhs.norm("l2")
        



# problem should be a NonlinearVariationalProblem
def CustomNonlinearSolver_old(problem, u = None, du = None, bcs = None, callbacks = [], Nitermax = 10, tol = 1e-8): 
    # u should be provided
    if(isinstance(problem, CustomNonlinearProblem) and u):
        Jac = problem.a
        Res = problem.L
        bcs = problem.bcs
        u = u

    # bcs should be provided
    elif(isinstance(problem, df.NonlinearVariationalProblem) and bcs):
        Jac = problem.J_ufl
        Res = problem.F_ufl
        u = problem.u_ufl
        bcs = bcs
        
    return Newton(Jac, Res, bcs, du, u, callbacks, Nitermax, tol)


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

    def solve_subproblem(self, i):   
        df.assemble(self.subproblems[i].L_ufl, tensor = self.F[i])    
        [bc.apply(self.F[i]) for bc in self.subproblems[i].bcs()]
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
    
    V = u.function_space()
    du.vector().set_local(np.zeros(V.dim()))
    [foo(u,du) for foo in callbacks]
    
    if(u0_satisfybc):
        for bc_i in bc: # non-homogeneous dirichlet applied only in the first itereation
            bc_i.homogenize()
        
    A, b = df.assemble_system(Jac, -Res, bc)
    
    nRes = []
    nRes.append(b.norm("l2"))
    nRes[0] = nRes[0] if nRes[0]>0.0 else 1.0
    
    niter = 0
    for bc_i in bc: # non-homogeneous dirichlet applied only in the first itereation
        bc_i.homogenize()
    
    while nRes[niter]/nRes[0] > tol and niter < Nitermax:
        df.solve(A, du.vector(), b)
        u.assign(u + du)    
        [foo(u,du) for foo in callbacks]
            
        A, b = df.assemble_system(Jac, -Res, bc)
        nRes.append(b.norm("l2"))
        print(" Residual:", nRes[niter+1])
        niter += 1
    
    return u, nRes



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
    def __init__(self, V, dx, sol = None, inner_representation = 'quadrature', outer_representation = 'uflacs'):    
        
        self.V = V
        self.sol = sol if sol else df.Function(V)
        self.dx = dx
        self.inner_representation = inner_representation
        self.outer_representation = outer_representation
        
        df.parameters["form_compiler"]["representation"] = self.inner_representation
        
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        
        a_proj = df.inner(u, v)*self.dx
        self.b_proj = lambda w: df.inner(w, v)*self.dx
        
        self.rhs = df.PETScVector()
        
        self.solver = df.LocalSolver(a_proj)
        self.solver.factorize()
        
        df.parameters["form_compiler"]["representation"] = self.outer_representation

    
    def __call__(self, u):
        df.parameters["form_compiler"]["representation"] = self.inner_representation
        df.assemble(self.b_proj(u), tensor = self.rhs)
        self.solver.solve_local(self.sol.vector(), self.rhs,  self.V.dofmap())
        df.parameters["form_compiler"]["representation"] = self.outer_representation
        
# Class for interfacing with the Newton solver
class myNonlinearProblem(df.NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)