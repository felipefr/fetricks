"""
This file is part of fetricksx:  useful tricks and some extensions for FEniCsX and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM and FEniCsX ).

Copyright (c) 2022-2024, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or
<f.rocha.felipe@gmail.com>
"""
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, la 
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
        x0 = [] if u is None else [u.x]
        fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=x0)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        x0 = None if u is None else u.x
        fem.petsc.set_bc(self._b, self.bcs, x0)

    def assemble_lhs(self):
        self._A.zeroEntries()
        fem.petsc.assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

    def solve_system(self):
        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()

# problem should be a NonlinearVariationalProblem

class CustomNonlinearProblem:
    def __init__(self, res, u, bcs, jac):
        self.res = res
        self.u = u
        self.bcs = bcs
        self.jac = jac
        

        

class CustomNonlinearSolver:

    # bcs: original object
    def __init__(self, problem, callbacks = [], u0_satisfybc = False): 
        
        self.problem = problem
        self.callbacks = callbacks
        self.u0_satisfybc = u0_satisfybc
    
        self.du = fem.Function(self.problem.u.function_space)
 
        self.tangent_problem = CustomTangentProblem(
        self.problem.jac, -self.problem.res,
        u=self.du,
        bcs=self.problem.bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "petsc"}) # seems faster than mumps
        
    def reset_bcs(self, bcs):
        self.problem.reset_bcs(bcs)
        
    def solve(self, Nitermax = 10, tol = 1e-8, report = False):
        # compute the residual norm at the beginning of the load step
        self.call_callbacks()
        nRes = []
        
        self.tangent_problem.assemble_rhs()
        nRes.append(self.tangent_problem._b.norm())
        if(nRes[0]<tol): 
            nRes[0] = 1.0
        self.du.x.array[:] = 0.0

        niter = 0
        while nRes[niter] / nRes[0] > tol and niter < Nitermax:
            # solve for the displacement correction
            self.tangent_problem.assemble_lhs()
            self.tangent_problem.solve_system()

            # update the displacement increment with the current correction
            self.problem.u.x.petsc_vec.axpy(1, self.du.x.petsc_vec)  # Du = Du + 1*du
            self.problem.u.x.scatter_forward()
            self.call_callbacks()
            
            self.tangent_problem.assemble_rhs()
            nRes.append(self.tangent_problem._b.norm())
            
            niter += 1
            if(report):
                print(" Residual:", nRes[-1]/nRes[0])

        return nRes

    def call_callbacks(self):
        [foo(self.problem.u, self.du) for foo in self.callbacks]



class BlockSolver:
    def __init__(self, lhs, rhs, sol, bcs):
        self.n_subproblems = len(rhs)        
        
        if(isinstance(lhs, list)):    
            self.list_solver = [CustomLinearSolver(lhs[i], rhs[i], sol[i], bcs[i]) 
                                for i in range(self.n_subproblems)]
        
        else:
            self.list_solver = [CustomLinearSolver(lhs, rhs[0], sol[0], bcs[0])] 
            self.list_solver += [CustomLinearSolver(self.list_solver[0].lhs, rhs[i], sol[i], 
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



class Nonlinear_SNESProblem:
    def __init__(self, F, u, bcs, jac):
        self.L = fem.form(F)
        self.a = fem.form(jac)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
            
        fem.petsc.assemble_vector(F, self.L)
        
        fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        J.zeroEntries()
        fem.petsc.assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()


class Nonlinear_SNESSolver:
    def __init__(self, problem, u):
        V = u.function_space
        self.b = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
        self.J = fem.petsc.create_matrix(problem.a)
        
        # Create Newton solver and solve
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(problem.F, self.b)
        self.snes.setJacobian(problem.J, self.J)
        
        self.snes.setTolerances(rtol=1.0e-9, max_it=10)
        self.snes.getKSP().setType("preonly")
        self.snes.getKSP().setTolerances(rtol=1.0e-9)
        self.snes.getKSP().getPC().setType("lu")
        
        # For SNES line search to function correctly it is necessary that the
        # u.x.petsc_vec in the Jacobian and residual is *not* passed to
        # snes.solve.
        self.x = u.x.petsc_vec.copy()
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    def solve(self):
        self.snes.solve(None, self.x)
    
    def destroy(self):
        self.b.destroy()
        self.J.destroy()
        self.snes.destroy()


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