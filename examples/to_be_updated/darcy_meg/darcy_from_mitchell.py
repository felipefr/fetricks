import numpy
from dolfin import *
from petsc4py import PETSc
import time
from timeit import default_timer as timer
import numpy as np

def darcy(mesh):
    # "Mixed H(div) x L^2 formulation of Poisson/Darcy."
    
    V = FiniteElement("RT", mesh.ufl_cell(), 1)
    Q = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, V*Q)
    
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    a = (dot(u, v) + div(u)*q + div(v)*p)*dx
    f = Constant(-1.0)
    L = f*q*dx 

    A = assemble(a)
    b = assemble(L)
    w = Function(W)

    return A, b, w


def set_petscopt(prefix):
    PETScOptions.set(prefix + "ksp_type", "gmres")
    PETScOptions.set(prefix + "pc_type", "fieldsplit")
    PETScOptions.set(prefix + "pc_fieldsplit_type", "schur")
    PETScOptions.set(prefix + "pc_fieldsplit_schur_fact_type", "full")

    PETScOptions.set(prefix + "fieldsplit_0_ksp_type", "cg")
    PETScOptions.set(prefix + "fieldsplit_0_pc_type", "ilu")
    PETScOptions.set(prefix + "fieldsplit_0_ksp_rtol", 1.e-12)

    PETScOptions.set(prefix + "fieldsplit_1_ksp_type", "cg")
    PETScOptions.set(prefix + "fieldsplit_1_pc_type", "none")
    PETScOptions.set(prefix + "fieldsplit_1_ksp_rtol", 1.e-12)
    

def set_petscopt_2(prefix):
    PETScOptions.set(prefix + "ksp_type", "gmres")
    PETScOptions.set(prefix + "pc_type", "ilu")


def create_fields_split(w):
    W = w.function_space()
    u_dofs = W.sub(0).dofmap().dofs()
    p_dofs = W.sub(1).dofmap().dofs()
    u_is = PETSc.IS().createGeneral(u_dofs)
    p_is = PETSc.IS().createGeneral(p_dofs)
    fields = [("0", u_is), ("1", p_is)]
    
    return fields

def set_petscsolver(A, prefix, fields_split = None):
    solver = PETScKrylovSolver() # Will be overwritten
    solver.parameters["error_on_nonconvergence"] = True
    solver.parameters["relative_tolerance"] = 1.e-10
    solver.parameters["convergence_norm_type"] = "preconditioned"
    solver.parameters["monitor_convergence"] = True
    solver.parameters["report"] = True

    solver.set_operator(A)

    # Extrct the KSP (Krylov Solver P) from the solver
    ksp = solver.ksp()
    ksp.setOptionsPrefix(prefix)
    ksp.setFromOptions()
    
    if(fields_split):
        ksp.pc.setFieldSplitIS(*fields_split)
    
    return solver

def solve_darcy(A, b, w, prefix, fields_split = None):

    solver = set_petscsolver(A, prefix, fields_split)
    iterations = solver.solve(w.vector(), b)
    print("#iterations = ", iterations)

    return w
    
if __name__ == "__main__":

    n = 16
    mesh = UnitCubeMesh(n, n, n)
    R = 10
    
    
    
    prefix = "darcy_"
    set_petscopt(prefix)
    
    prefix2 = "standard_"
    set_petscopt_2(prefix)
    
    
    A, b, w = darcy(mesh)
    fields = create_fields_split(w)
    
    
    start = timer()
    
    for i in range(R):
        w = solve_darcy(A, b, w, prefix2, fields)
        
    end = timer()
    print(end - start)
    (u, p) = w.split(deepcopy=True)
    print(np.mean(p.vector().get_local()))
    # plot(u)
    # plot(p)
    # interactive()
