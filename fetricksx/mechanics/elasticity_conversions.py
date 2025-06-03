"""
This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import numpy as np

# doc missing
def convertParam(param,foo):
    
    n = len(param)
    paramNew = np.zeros((n,2))
    for i in range(n):
        paramNew[i,0], paramNew[i,1] = foo( *param[i,:].tolist()) 
  
    return paramNew

# doc missing
convertParam2 = lambda p,f: np.array( [  f(*p_i) for p_i in p ] )

def youngPoisson2lame_planeStress(nu,E):
    lamb , mu = youngPoisson2lame(nu,E)
    lamb = (2.0*mu*lamb)/(lamb + 2.0*mu)    
    return lamb, mu


def get_Celas_mandel(param, model = 'isotropic', gdim = 2, is_axisymmetric = False):
    if(model == 'isotropic'):
        if 'lamb' in param.keys():
            lamb , mu = param['lamb'], param['mu']
        elif 'E' in param.keys():
            E, nu = param['E'], param['nu']
            lamb, mu = youngPoisson2lame(nu,E)    
        else:
            print("provide a valid set of parameters (E,nu) or (lamb, mu)")

        if(is_axisymmetric):
            return np.array( [[lamb + 2*mu, lamb, lamb, 0], 
                              [lamb, lamb + 2*mu, lamb, 0],
                              [lamb, lamb, lamb + 2*mu, 0],
                              [0,    0,     0,       2*mu]] )
        else:
            return np.array( [[lamb + 2*mu, lamb, 0], [lamb, lamb + 2*mu, 0], [0, 0, 2*mu]] )
    
    elif(model == 'trans_isotropic' and is_axisymmetric):
        E1, E3 = param['E1'], param['E3']
        nu12, nu13, G13 = param['nu12'], param['nu13'], param['G13']
        S = np.array( [[1/E1, -nu12/E1, -nu13/E3, 0], 
                       [-nu12/E1, 1/E1, -nu13/E3, 0],
                       [-nu13/E3, -nu13/E3, 1/E3, 0],
                       [0,       0,       0,  1/(2*G13)]] )
    
        return np.linalg.inv(S)
    
    else:
        print("provide a valid model combination: 'isotropic', 'trans_isotropic'")

# Does not enforce symmetry
def get_Celas_flat(lamb,mu):
    return np.array( [[lamb + 2*mu, 0, 0, lamb], [0, 2*mu, 0, 0], [0, 0, 2*mu, 0], [lamb, 0, 0, lamb + 2*mu]] )

youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ] # lamb, mu

gof = lambda g,f: lambda x,y : g(*f(x,y)) # composition, g : R2 -> R* , f : R2 -> R2

lame2youngPoisson  = lambda lamb, mu : [ 0.5*lamb/(mu + lamb) , mu*(3.*lamb + 2.*mu)/(lamb + mu) ]
youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ]

# Star means plane strain/stress conversion
lame2lameStar = lambda lamb, mu: [(2.0*mu*lamb)/(lamb + 2.0*mu), mu]
lameStar2lame = lambda lambStar, mu: [(2.0*mu*lambStar)/(-lambStar + 2.0*mu), mu]

eng2lamb = lambda nu, E: nu * E/((1. - 2.*nu)*(1.+nu))
eng2mu = lambda nu, E: E/(2.*(1. + nu))
lame2poisson = lambda lamb, mu: 0.5*lamb/(mu + lamb)
lame2young = lambda lamb, mu: mu*(3.*lamb + 2.*mu)/(lamb + mu)
lame2lambPlaneStress = lambda lamb, mu: (2.0*mu*lamb)/(lamb + 2.0*mu)
lamePlaneStress2lamb = lambda lambStar, mu: (2.0*mu*lambStar)/(-lambStar + 2.0*mu)
eng2lambPlaneStress = gof(lame2lambPlaneStress,lambda x,y: (eng2lamb(x,y), eng2mu(x,y)))

