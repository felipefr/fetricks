#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 20:17:35 2022

@author: felipe
"""

import numpy as np


from ddfenics.mechanics.misc import eng2mu, eng2lamb, eng2lambPlane
from ddfenics.fenics.wrapper_expression import getMyCoeff

def getLameInclusions(nu1,E1,nu2,E2,M, op='cpp'):
    mu1 = eng2mu(nu1,E1)
    lamb1 = eng2lamb(nu1,E1)
    mu2 = eng2mu(nu2,E2)
    lamb2 = eng2lamb(nu2,E2)

    param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1,mu1], [lamb2,mu2]])
    
    print(param)
    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = getMyCoeff(materials, param, op = op)
    
    return lame

