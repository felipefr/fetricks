import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
from fetricks.fenics.mesh.mesh import Mesh 

import fetricks as ft 


from timeit import default_timer as timer

from functools import partial 

df.parameters["form_compiler"]["representation"] = 'uflacs'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

ppos = lambda x: (x+abs(x))/2.


class materialModel:
    
    def sigma(self,eps_el):
        pass
    
    def tangent(self, e):
        pass
    
    def createInternalVariables(self, W, W0):
        pass
    
    def update_alpha(self,deps, old_sig, old_p):
        pass
    
    def project_var(self,AA, dxm):
        for label in AA.keys(): 
            ft.local_project_given_sol(AA[label], self.varInt[label].function_space(), self.varInt[label], dxm)

class hyperlasticityModel(materialModel):
    
    def __init__(self,E,nu, alpha):
        self.lamb = df.Constant(E*nu/(1+nu)/(1-2*nu))
        self.mu = df.Constant(E/2./(1+nu))
        self.alpha = df.Constant(alpha)
        
    def createInternalVariables(self, W, W0):
        self.sig = df.Function(W)
        self.eps = df.Function(W)
        self.tre2 = df.Function(W0)
        self.ee = df.Function(W0)
    
        self.varInt = {'tre2': self.tre2, 'ee' : self.ee, 'eps' : self.eps,  'sig' : self.sig} 

    def sigma(self, lamb_, mu_, eps): # elastic (I dont know why for the moment) # in mandel format
        return lamb_*ft.tr_mandel(eps)*ft.Id_mandel_df + 2*mu_*eps
    
    def epseps_e(self, de):
        return df.inner(self.eps, de)*self.eps

    
    def tangent(self, de):
        lamb_ = self.lamb*( 1 + 3*self.alpha*self.tre2)
        mu_ = self.mu*( 1 + self.alpha*self.ee ) 
        
        de_mandel = ft.tensor2mandel(de)
        
        return self.sigma(lamb_, mu_, de_mandel)  + 4*self.mu*self.alpha*self.epseps_e(de_mandel)

    def update_alpha(self, eps_new, dxm):
        
        ee = df.inner(eps_new,eps_new)
        tre2 = ft.tr_mandel(eps_new)**2.0
        
        lamb_ = self.lamb*( 1 + self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        alpha_new = {'tre2': tre2, 'ee' : ee, 'eps' : eps_new, 'sig': self.sigma(lamb_, mu_, eps_new)}
        self.project_var(alpha_new, dxm)


class hyperlasticityModel_simple(materialModel):
    
    def __init__(self,E,nu, alpha):
        self.lamb = df.Constant(E*nu/(1+nu)/(1-2*nu))
        self.mu = df.Constant(E/2./(1+nu))
        self.alpha = df.Constant(alpha)
        
    def createInternalVariables(self, W, W0):
        self.sig = df.Function(W)
        self.eps = df.Function(W)
    
        self.varInt = {'eps' : self.eps,  'sig' : self.sig} 

    def sigma(self, lamb_, mu_, eps): # elastic (I dont know why for the moment) # in mandel format
        return lamb_*ft.tr_mandel(eps)*ft.Id_mandel + 2*mu_*eps
    
    def epseps_e(self, de):
        return df.inner(self.eps, de)*self.eps

    
    def tangent(self, de):
        ee = df.inner(self.eps, self.eps)
        tre2 = ft.tr_mandel(self.eps)**2.0
        
        lamb_ = self.lamb*( 1 + 3*self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        de_mandel = ft.tensor2mandel(de)
        
        return self.sigma(lamb_, mu_, de_mandel)  + 4*self.mu*self.alpha*self.epseps_e(de_mandel)

    def update_alpha(self, eps_new):
        
        ee = df.inner(eps_new,eps_new)
        tre2 = ft.tr_mandel(eps_new)**2.0
        
        lamb_ = self.lamb*( 1 + self.alpha*tre2)
        mu_ = self.mu*( 1 + self.alpha*ee ) 
        
        alpha_new = {'eps' : eps_new, 'sig': self.sigma(lamb_, mu_, eps_new)}
        self.project_var(alpha_new)
    

class sigmaExpression(df.UserExpression):
    def __init__(self, eps, sigma_law,  **kwargs):
        self.eps = eps
        self.sig_law = sigma_law
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        eps_ = self.eps.vector().get_local()[cell.index*3:(cell.index + 1)*3]
        values[:] = self.sig_law(eps_)

    def value_shape(self):
        return (3,)



class tangentExpression(df.UserExpression):
    def __init__(self, eps, tangent_law,  **kwargs):
        self.eps = eps
        self.tangent_law = tangent_law
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        eps_ = self.eps.vector().get_local()[cell.index*3:(cell.index + 1)*3]
        values[:] = self.tangent_law(eps_).flatten()

    def value_shape(self):
        return (3,3,)

   
# elastic parameters

E = 100.0
nu = 0.3
alpha = 200.0
ty = 5.0

model = hyperlasticityModel(E, nu, alpha)

mesh = Mesh("./meshes/mesh_40.xdmf")

start = timer()

clampedBndFlag = 2 
LoadBndFlag = 1 
traction = df.Constant((0.0,ty ))
    
deg_u = 1
deg_stress = 0
V = df.VectorFunctionSpace(mesh, "CG", deg_u)
We = df.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=3, quad_scheme='default')
W = df.FunctionSpace(mesh, We)
W0e = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = df.FunctionSpace(mesh, W0e)

bcL = df.DirichletBC(V, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
bc = [bcL]

def F_ext(v):
    return df.inner(traction, v)*mesh.ds(LoadBndFlag)

model.createInternalVariables(W, W0)
u = df.Function(V, name="Total displacement")
du = df.Function(V, name="Iteration correction")
v = df.TestFunction(V)
u_ = df.TrialFunction(V)


metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = df.dx(metadata=metadata)

# alpha_ = update_sigma(deps, sig_old, p)

def sigma_law(e, lamb, mu, alpha): # elastic (I dont know why for the moment) # in mandel format

    ee = np.dot(e,e)
    tre2 = (e[0] + e[1])**2.0
    
    lamb_ = lamb*( 1 + alpha*tre2)
    mu_ = mu*( 1 + alpha*ee ) 
    
    return lamb_*(e[0] + e[1])*ft.Id_mandel_np + 2*mu_*e


def tangent_law(e, lamb, mu, alpha): # elastic (I dont know why for the moment) # in mandel format
    
    ee = np.dot(e,e)
    tre2 = (e[0] + e[1])**2.0
    
    lamb_ = lamb*( 1 + 3*alpha*tre2)
    mu_ = mu*( 1 + alpha*ee ) 
    
    D = 4*mu*alpha*np.outer(e,e)

    D[0,0] += lamb_ + 2*mu_
    D[1,1] += lamb_ + 2*mu_
    D[0,1] += lamb_
    D[1,0] += lamb_
    D[2,2] += 2*mu_
    
    return D


sigma_law_ = partial(sigma_law, lamb = 57.692307692307686, mu = 38.46153846153846, alpha = 200.0 )
tangent_law_ = partial(tangent_law, lamb = 57.692307692307686, mu = 38.46153846153846, alpha = 200.0 )

sig_expr = sigmaExpression(model.eps, sigma_law_)
tang_expr = tangentExpression(model.eps, tangent_law_)

# sig_expr = sigma_law_(eps_expr) 

a_Newton = df.inner(ft.tensor2mandel(ft.symgrad(u_)), df.dot(tang_expr, ft.tensor2mandel(ft.symgrad(v))) )*dxm
res = -df.inner(ft.tensor2mandel(ft.symgrad(v)), sig_expr )*dxm + F_ext(v)

file_results = df.XDMFFile("cook.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


callbacks = [lambda w: model.update_alpha(ft.tensor2mandel(ft.symgrad(w)), dxm) ]

ft.Newton(a_Newton, res, bc, du, u, callbacks , Nitermax = 10, tol = 1e-8)

## Solve here Newton Raphson

file_results.write(u, 0.0)

end = timer()
print(end - start)


# plt.plot(results[:, 0], results[:, 1], "-o")
# plt.xlabel("Displacement of inner boundary")
# plt.ylabel(r"Applied pressure $q/q_{lim}$")
# plt.show()
