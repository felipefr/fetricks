from dolfin import *
import numpy
from timeit import default_timer as timer
import numpy as np

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

def primal(mesh, tag):
    #"Standard H^1(mesh) formulation of Poisson's equation." 
    
    Q = FunctionSpace(mesh, "CG", 1)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    a = inner(grad(p), grad(q))*dx
    f = Constant(1.0)
    L = f*q*dx

    bc = DirichletBC(Q, 0.0, "on_boundary")

    A, b = assemble_system(a, L, bc)

    p = Function(Q)
    return (A, b, p)

def primal_lu(mesh, tag):
    #"Solve primal H^1 formulation using LU."

    A, b, p = primal(mesh, tag)
    solver = LUSolver(A)
    solver.solve(p.vector(), b)

    return p

def primal_amg(mesh, tag):
    #"Solve primal H^1 formulation using CG with AMG."

    A, b, p = primal(mesh, tag)

    solver = PETScKrylovSolver("cg", "amg")
    solver.set_operator(A)
    num_it = solver.solve(p.vector(), b)

    print("%s: num_it = " % tag, num_it)
    return p

def darcy(mesh):
    #"Mixed H(div) x L^2 formulation of Poisson/Darcy."
    
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
    
def darcy_lu(mesh, tag):
    #"Solve mixed H(div) x L^2 formulation using LU"
    
    (A, b, w) = darcy(mesh)
    solve(A, w.vector(), b)
    
    return w

def darcy_prec1(W):
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    prec = (inner(u, v) + div(u)*div(v) + p*q)*dx
    B = assemble(prec)
    return B

def darcy_amg(mesh, A, b, w, B = None):
    #"Solve mixed H(div) x L^2 formulation using AMG"
    
    # B = darcy_prec1(w.function_space())
    solver = PETScKrylovSolver("gmres", "ilu")
    solver.set_operators(A, B)
    solver.parameters["monitor_convergence"] = True
    solver.parameters["report"] = True
    solver.parameters["relative_tolerance"] = 1.e-10
    num_iter = solver.solve(w.vector(), b)
    #solve(A, w.vector(), b)
    print("num_iter = ", num_iter)
    
    # (u, p) = w.split(deepcopy=True)
    # plot(u)
    # plot(p)
    # interactive()

    return w

def darcy_ilu(mesh, A, b, w):
    #"Solve mixed H(div) x L^2 formulation using iLU"
    
    solver = PETScKrylovSolver("gmres", "ilu")
    solver.set_operator(A)
    solver.parameters["monitor_convergence"] = True
    solver.parameters["report"] = True
    solver.parameters["relative_tolerance"] = 1.e-10
    num_iter = solver.solve(w.vector(), b)
    print("num_iter = ", num_iter)
    
    # (u, p) = w.split(deepcopy=True)
    # plot(u)
    # plot(p)
    # interactive()

    return w

def time_solve(mesh, algorithm, tag):

    solution = algorithm(mesh, tag)
    times = timings(TimingClear_clear, [TimingType_wall])
    dim = solution.function_space().dim()
    t = times.get_value(tag, "wall tot")

    return (t, dim)

def time_solves(mesh, algorithm, tag, R=1):

    times = numpy.empty(R)
    h = mesh.hmax()
    for i in range(R):
        t, dim = time_solve(mesh, algorithm, tag)
        print("%s (s) with N=%d and h=%.2g: %.3g" % (tag, dim, h, t))
        times[i] = t

    avg_t = numpy.mean(times)
    std_t = numpy.std(times)

    return (avg_t, std_t)
        
if __name__ == "__main__":

    #set_log_level(ERROR)
    
    n = 16
    mesh = UnitCubeMesh(n, n, n)
    h = mesh.hmax()
    
    # Number of repetitions to do timings statistics on
    R = 10
    start = timer()
    for i in range(R):
        (A, b, w) = darcy(mesh)
        B = darcy_prec1(w.function_space())
        darcy_ilu(mesh, A, b, w)
    end = timer()
    
    print(end - start)
    
    
    (u, p) = w.split(deepcopy=True)
    print(np.mean(p.vector().get_local()))
    
    #tag = "Primal solve: amg"
    #avg_t, std_t = time_solves(mesh, primal_amg, tag, R=R)
    #print "%s took %0.3g (+- %0.3g)" % (tag, avg_t, std_t)
    #print

    # tag = "Darcy solve: amg"
    # avg_t, std_t = time_solves(mesh, darcy_amg, tag, R=R)
    # print("%s took %0.3g (+-) %0.3g" % (tag, avg_t, std_t))

    #(u, p) = w.split(deepcopy=True)
    #plot(u)
    #plot(p)
    #interactive()
