import dolfin as df
import numpy as np
from ufl import nabla_div

from deepBND.core.fenics_tools.misc import symgrad
import deepBND.core.fenics_tools.misc as misc
from deepBND.core.elasticity.misc import eng2mu, eng2lambPlane
from deepBND.core.fenics_tools.wrapper_expression import getMyCoeff

halfsq2 = np.sqrt(2.)/2. # useful for mandel notation

# Voigt notation for strain (for stress becomes only sigma_12 at third entry)
symgrad_voigt = lambda v: df.as_vector([v[0].dx(0), 
                                        v[1].dx(1), 
                                        v[0].dx(1) + v[1].dx(0) ])

# Mandel-Kelvin notation for both strain and stress
symgrad_mandel = lambda v: df.as_vector([v[0].dx(0), 
                                         v[1].dx(1), 
                                         halfsq2*(v[0].dx(1) + v[1].dx(0)) ])

def sigmaLame(u, lame):
    return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*symgrad(u)

def vonMises(sig):
    s = sig - (1./3)*df.tr(sig)*df.Identity(2)
    return df.sqrt((3./2)*df.inner(s, s)) 


def getLameInclusions(nu1,E1,nu2,E2,M, op='cpp'):
    mu1 = eng2mu(nu1,E1)
    lamb1 = eng2lambPlane(nu1,E1)
    mu2 = eng2mu(nu2,E2)
    lamb2 = eng2lambPlane(nu2,E2)
    param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1,mu1], [lamb2,mu2]])
    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = getMyCoeff(materials, param, op = op)
    
    return lame

# Piola transform used to rotate solutions
def PiolaTransform_rotation(theta, Vref): #only detB = pm 1 in our context
    Mref = Vref.mesh()    
    
    s = np.sin(theta) ; c = np.cos(theta)
    B = np.array([[c,-s],[s,c]])

    Finv = misc.affineTransformationExpression(np.zeros(2), B.T, Mref)
    Bmultiplication = misc.affineTransformationExpression(np.zeros(2), B, Mref)
    
    return lambda sol: df.interpolate( misc.myfog_expression(Bmultiplication, misc.myfog(sol,Finv)), Vref) #


def PiolaTransform_rotation_matricial(theta, Vref): #only detB = pm 1 in our context
    Nh = Vref.dim()    
    Piola = PiolaTransform_rotation(theta,Vref)
    Pmat = np.zeros((Nh,Nh))
    
    phi_j = df.Function(Vref)
    ej = np.zeros(Nh) 
    
    for j in range(Nh):
        ej[j] = 1.0
        phi_j.vector().set_local(ej)
        Pmat[:,j] = Piola(phi_j).vector().get_local()[:]
        ej[j] = 0.0
        
    return Pmat