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

class BlockSolver:
    def __init__(self, lhs_form, list_rhs_form, list_sol, list_bcs):
        self.list_rhs_form = list_rhs_form
        self.list_sol = list_sol
        self.list_bcs = list_bcs
        self.n_subproblems = len(list_rhs_form)
        domain = self.list_sol[0].function_space.mesh
        
        # generate associated matrix
        self.lhs_form = fem.form(lhs_form)
        A = fem.petsc.assemble_matrix(self.lhs_form,bcs=self.list_bcs[0])
        A.assemble()

        self.solver = PETSc.KSP().create(domain.comm)
        self.solver.setOperators(A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        
        self.list_rhs_form = [fem.form(b) for b in list_rhs_form]
        self.b = [fem.petsc.create_vector(b) for b in self.list_rhs_form]
     
    def assembly_rhs(self):
       for i in range(self.n_subproblems):
           with self.b[i].localForm() as b:
               b.set(0.0)
           
           fem.petsc.assemble_vector(self.b[i],self.list_rhs_form[i])
           fem.petsc.apply_lifting(self.b[i], [self.lhs_form],[self.list_bcs[i]])
           self.b[i].ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)    
           fem.petsc.set_bc(self.b[i],self.list_bcs[i])

    def solve(self):  
       self.assembly_rhs()
       for i in range(self.n_subproblems):
           self.solver.solve(self.b[i],self.list_sol[i].vector)
           self.list_sol[i].x.scatter_forward()


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