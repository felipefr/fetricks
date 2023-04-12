#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:23:02 2023

@author: ffiguere
"""
from sympy import *

I1, I2, I3 = symbols("I1 I2 I3")
I1bar = I3**(-1/3)*I1
I2bar = I3**(-2/3)*I2
J = I3**(1/2)

def mooney_rivlin():
    c1, c2 = symbols("c1 c2")
    return c1*(I1bar - 3) + c2*(I2bar - 3)


def hartmann_neff():
    alpha, c10, c01, kappa = symbols("alpha c10 c01 kappa")
    
    U = (kappa/50)*(J**5 + J**(-5) - 2)
    W = alpha*(I1bar**3 - 27) + c10*(I1bar - 3) + c01*(I2bar**1.5 - 3.*sqrt(3))
    
    return W + U


psi = hartmann_neff()

dpsi = [diff(psi,Ii) for Ii in [I1,I2,I3]]
d2psi = [[diff(dpsi[j],Ii) for Ii in [I1,I2,I3]] for j in range(3)]

for i in range(3):
    print("dpsi{0} =".format(i+1), nsimplify(dpsi[i]))
    

for i in range(3):
    for j in range(i,3):
        print("d2psi{0}{1} =".format(i+1,j+1), nsimplify(d2psi[i][j]))