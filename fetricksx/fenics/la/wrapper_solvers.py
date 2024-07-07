"""
This file is part of fetricksx:  useful tricks and some extensions for FEniCsX and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM and FEniCsX ).

Copyright (c) 2022-2024, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or
<f.rocha.felipe@gmail.com>
"""
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from scipy.sparse import csr_matrix 

# petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

# essentially used to solve several times with the same LHS
class CustomLinearSolver:
    def __init__(self, lhs, rhs, sol, bcs, solver = None):
        self.sol = sol
        self.bcs = bcs
        domain = self.sol.function_space.mesh
        
        if(solver): #if solver is given
            self.solver = solver
            self.lhs = lhs
        else:
            self.solver = PETSc.KSP().create(domain.comm)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
            self.solver.getPC().setFactorSolverType("mumps")
            self.lhs = fem.form(lhs)
            self.assembly_lhs()
            
        self.rhs = fem.form(rhs)
        self.b = fem.petsc.create_vector(self.rhs)
    
    def assembly_lhs(self):
        self.A = fem.petsc.assemble_matrix(self.lhs, bcs=self.bcs)
        self.A.assemble()
        self.solver.setOperators(self.A)

    def assembly_rhs(self):
        with self.b.localForm() as b:
            b.set(0.0)
        
        fem.petsc.assemble_vector(self.b, self.rhs)
        fem.petsc.apply_lifting(self.b, [self.lhs], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)    
        fem.petsc.set_bc(self.b,self.bcs)

    def solve(self):  
       self.assembly_rhs()
       self.solver.solve(self.b,self.sol.vector)
       self.sol.x.scatter_forward()


# Based on https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/plasticity/plasticity.html
class CustomTangentProblem(fem.petsc.LinearProblem):
    def assemble_rhs(self, u=None):
        """Assemble right-hand side and lift Dirichlet bcs.

        Parameters
        ----------
        u : dolfinx.fem.Function, optional
            For non-zero Dirichlet bcs u_D, use this function to assemble rhs with the value u_D - u_{bc}
            where u_{bc} is the value of the given u at the corresponding. Typically used for custom Newton methods
            with non-zero Dirichlet bcs.
        """

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        x0 = [] if u is None else [u.vector]
        fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=x0, scale=1.0)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        x0 = None if u is None else u.vector
        fem.petsc.set_bc(self._b, self.bcs, x0, scale=1.0)

    def assemble_lhs(self):
        self._A.zeroEntries()
        fem.petsc.assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

    def solve_system(self):
        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()
        
class CustomNonlinearSolver:
    
    tangent_problem = CustomLinearProblem(
    tangent_form,
    -Residual,
    u=du,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)


# problem should be a NonlinearVariationalProblem
class CustomNonlinearSolver:

    # bcs: original object
    def __init__(self, problem, callbacks = [], u0_satisfybc = False): 
        
        self.problem = problem
        self.callbacks = callbacks
        self.u0_satisfybc = u0_satisfybc
    
        self.du = df.Function(self.problem.u.function_space())
        self.du.vector().set_local(np.zeros(self.du.vector().size()))
        
        self.rhs = df.PETScVector() 
        self.lhs = df.PETScMatrix()
        
    def reset_bcs(self, bcs):
        self.problem.reset_bcs(bcs)
        
    def solve(self, Nitermax = 10, tol = 1e-8, report = True):
        
        if(self.u0_satisfybc): self.homogenise_bcs()
        self.call_callbacks()
        nRes = []
        
        # iteration 1
        nRes.append(self.assemble())

        self.homogenise_bcs() # after first iteration, u satisfies bc
                        
        nRes[0] = nRes[0] if nRes[0]>0.0 else 1.0
        niter = 0
        
        while nRes[niter]/nRes[0] > tol and niter < Nitermax:
            nRes.append(self.newton_raphson_iteration())
            niter+=1
            
            if(report):
                print(" Residual:", nRes[-1]/nRes[0])
            

        return nRes

    def homogenise_bcs(self):
        for bc_i in self.problem.bcs: # non-homogeneous dirichlet applied only in the first itereation
            bc_i.homogenize()
        
        
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

class BlockSolver:
    def __init__(self, lhs, rhs, sol, bcs):
        self.n_subproblems = len(rhs)        
        
        if(isinstance(lhs, list)):    
            self.list_solver = [custom_linear_solver(lhs[i], rhs[i], sol[i], bcs[i]) 
                                for i in range(self.n_subproblems)]
        
        else:
            self.list_solver = [custom_linear_solver(lhs, rhs[0], sol[0], bcs[0])] 
            self.list_solver += [custom_linear_solver(self.list_solver[0].lhs, rhs[i], sol[i], 
                                 bcs[i], solver = self.list_solver[0].solver) for i in range(1, self.n_subproblems)]                    

            
         
    def assembly_rhs(self):
       for i in range(self.n_subproblems):
           self.list_solver[i].assembly_rhs()

    def solve(self):  
       self.assembly_rhs()
       for i in range(self.n_subproblems):
           s = self.list_solver[i]
           s.solver.solve(s.b, s.sol.vector)
           s.sol.x.scatter_forward()


# Picard nonlinear iterations for mixed 
# q_k is the first solution (convention) in the previous step
# set sub if q_k is just a part subcomponent of w 
def picard(a, L, w, q_k, bcs, tol = 1.0e-6, maxiter = 25,  zerofy = True, sub = -1): 
    q_k.x.array[:] = 0.0    
    eps = 1.0           # error measure ||u-u_k||
    it = 0
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    problem = LinearProblem(a, L, bcs, w, petsc_options=petsc_options)
    while eps > tol and it < maxiter:
         it += 1
         problem.solve()
         if(sub>-1):
             q_new = w.sub(sub).collapse()         
         else: 
             q_new = w
         eps = np.linalg.norm(q_new.x.array - q_k.x.array, ord=np.Inf)/np.linalg.norm(q_new.x.array)
         print('iter=%d: norm=%g' % (it, eps))
         q_k.x.array[:] = q_new.x.array[:]   # update for next iteration
         

# From Dolfiny https://github.com/michalhabera/dolfiny/blob/master/dolfiny/la.py        
def petsc_to_scipy(A):
    """Converts PETSc serial matrix to SciPy CSR matrix"""
    ai, aj, av = A.getValuesCSR()
    mat = csr_matrix((av, aj, ai))

    return mat

# From Dolfiny https://github.com/michalhabera/dolfiny/blob/master/dolfiny/la.py   
def scipy_to_petsc(A):
    """Converts SciPy CSR matrix to PETSc serial matrix."""
    nrows = A.shape[0]
    ncols = A.shape[1]

    ai, aj, av = A.indptr, A.indices, A.data
    mat = PETSc.Mat()
    mat.createAIJ(size=(nrows, ncols))
    mat.setUp()
    mat.setValuesCSR(ai, aj, av)
    mat.assemble()

    return mat