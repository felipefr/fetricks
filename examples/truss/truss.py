import dolfin as df 
import numpy as np

import ddfenics.fenics.fenicsUtils as feut
import ddfenics.fenics.wrapper_io as iofe
# import ddfenics.core.fenics_tools.misc as feut
import ddfenics.mechanics.misc as mech

from ddfenics.dd.ddfunction import DDFunction

from functools import partial
from truss_utils import *


def solve_truss(mesh, sigma_law):
    Area = df.Constant(1.0) # assumes unitary area
    
    Lx, Ly = np.max(mesh.coordinates(), axis = 0)

    t = getTangentTruss(mesh)
    
    grad_truss = partial(tgrad, t = t)
    
    Ue = df.VectorElement("CG", mesh.ufl_cell(), 1, dim=2)
    Uh = df.FunctionSpace(mesh, Ue)

    se = df.VectorElement("DG", mesh.ufl_cell(), 0, dim = 2)
    sh0 = df.FunctionSpace(mesh, se)
 
    
    def cornerLeftBottom(x, on_boundary):
        return df.near(x[0], 0.0) and df.near(x[1], 0.0)

    def cornerLeftTop(x, on_boundary):
        return df.near(x[0], 0.0) and df.near(x[1], Ly)

    bc1 = df.DirichletBC(Uh, df.Constant((0.0,0.0)), cornerLeftBottom, method = 'pointwise' )
    bc2 = df.DirichletBC(Uh, df.Constant((0.0,0.0)), cornerLeftTop, method = 'pointwise')
    
    fy = -1.0
    p1 = df.Point((0.5*Lx, 0.0))
    p2 = df.Point((Lx, 0.0))             
    
    # # Define variational problem
    uh = df.TrialFunction(Uh) 
    vh = df.TestFunction(Uh)
    
    dx = df.Measure('dx', mesh)
    
    a = df.inner( Area*sigma_law(grad_truss(uh)),  grad_truss(vh))*dx
    b = df.inner( df.Constant((0.,0.)) , vh)*dx
    
    A, b = df.assemble_system(a, b, [bc1, bc2])
    
    for p in [p1, p2]: 
        delta = df.PointSource(Uh.sub(1), p, fy)
        delta.apply(b) # Only in b (I don't why)

    uh = df.Function(Uh)
    
    # Compute solution
    df.solve(A, uh.vector(), b)   
    

    g = grad_truss(uh)

    zh = DDFunction(sh0)
    zh.update(df.as_tensor((g,sigma_law(g))))
    
    
    return uh, zh

if __name__ == '__main__':
    
    mesh = getMeshTruss()
    
    E = 100.0
    sigma_law = lambda e: df.Constant(E)*e 
    
    uh, zh = solve_truss(mesh, sigma_law )

    # epsh.rename('eps', '')
    zh.rename('z', '')
    uh.rename('u', '')
    
    iofe.exportXDMF_gen("truss_vtk.xdmf", fields={'vertex': [uh], 'cell': [zh]})
    
    
    print(np.min(uh.vector().get_local()), np.max(uh.vector().get_local()))
    print(np.min(zh.data(), axis = 0))
    print(np.max(zh.data(), axis = 0))
    
    np.savetxt('database_ref.txt', zh.data(), header = '1.0 \n%d 2 1 1'%zh.data().shape[0], comments = '', fmt='%.8e', )




