#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 00:14:04 2022

@author: felipe


This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

import dolfin as df 
import numpy as np

import multiphenics as mp

from ddfenics.fenics.fenicsUtils import symgrad_voigt, symgrad
import ddfenics.fenics.fenicsUtils as feut
from ddfenics.fenics.wrapper_solvers import solver_direct
from ddfenics.fenics.enriched_mesh import EnrichedMesh 
import ddfenics.fenics.wrapper_io as iofe
# import ddfenics.core.fenics_tools.misc as feut
import ddfenics.mechanics.misc as mech

from ddfenics.dd.ddfunction import DDFunction

from mesh import CookMembrane
import generation_database

from functools import partial 

df.parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}

def solve_cook(meshfile, psi):

    mesh = EnrichedMesh(meshfile)

    Ue = df.VectorElement("CG", mesh.ufl_cell(), 1)
    Se = df.TensorElement("CG", mesh.ufl_cell(), 0, symmetry = True) 
    Me = df.MixedElement((Ue,Se,Se))
    Mh = df.FunctionSpace(mesh, Me)

    clampedBndFlag = 2 
    LoadBndFlag = 1 
    
    ty = 2.0
    traction = df.Constant((0.0,ty ))
    bcL = df.DirichletBC(Mh.sub(0), df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
        
    # Chom = Chom_multiscale(tangent_dataset, mapping, degree = 0)

    # # Define variational problem
    ues = df.Function(Mh)
    vues = df.TrialFunction(Mh)
    
    uh, epsh, sigh = ues.split()
    # vh, epsh, sigh = ues.split()
    
    
    # duh = df.TrialFunction(Uh)            # Incremental displacement
    # vh  = df.TestFunction(Uh)             # Test function
    # uh  = df.Function(Uh)                 # Displacement from previous iteration
    # epsh = df.Function(Sh)                 
    # sigh = df.Function(Sh)

    Pi = psi(epsh)*mesh.dx - df.inner(epsh - symgrad(uh), sigh)*mesh.dx - df.inner(traction,uh)*mesh.ds(LoadBndFlag)
     
    F = df.derivative(Pi, ues, vues)

    
    # J = df.derivative(F, uh, duh) # it will be computed even not providing it
    
    # # Compute solution
    df.solve(F==0, ues, bcL)
    
    
    # return uh


def solve_cook_multiphenics(meshfile, psi, sig):

    mesh = EnrichedMesh(meshfile)

    Ue = df.VectorFunctionSpace(mesh, "CG", 1)
    Se = df.TensorFunctionSpace(mesh, "DG", 0, symmetry = True) 
    Mh = mp.BlockFunctionSpace((Ue,Se,Se))

    clampedBndFlag = 2 
    LoadBndFlag = 1 
    
    ty = 2.0
    traction = df.Constant((0.0,ty ))
    bcL = mp.DirichletBC(Mh.sub(0), df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
    
    bc = mp.BlockDirichletBC([[bcL],[],[]])
        
    # Chom = Chom_multiscale(tangent_dataset, mapping, degree = 0)

    # # Define variational problem
    ues = mp.BlockFunction(Mh)
    vues = mp.BlockTrialFunction(Mh)
    dues = mp.BlockTestFunction(Mh)
    
    (uh, epsh, sigh) = mp.block_split(ues)
    (vh, vepsh, vsigh) = mp.block_split(vues)
    (duh, depsh, dsigh) = mp.block_split(dues)
    
    
    # duh = df.TrialFunction(Uh)            # Incremental displacement
    # vh  = df.TestFunction(Uh)             # Test function
    # uh  = df.Function(Uh)                 # Displacement from previous iteration
    # epsh = df.Function(Sh)                 
    # sigh = df.Function(Sh)


    l = lambda v: df.inner(traction, v)*mesh.ds(LoadBndFlag)   
    Pi = lambda e: psi(e)*mesh.dx 
    b = lambda s, t: df.inner(s, t)*mesh.dx    
        # df.inner(epsh - symgrad(uh), sigh)*mesh.dx - 
    
    
    F = [ b(sigh, symgrad(vh)) - l(vh),  
          b(sig(epsh), vepsh) - b(sigh, vepsh) , 
          b(epsh, vsigh) - b(symgrad(uh), vsigh) ] 

    print(F)
    J = mp.block_derivative(F, ues, dues)

    problem = mp.BlockNonlinearProblem(F, ues, bc, J)
    # solver = BlockPETScSNESSolver(problem)
    # solver.parameters.update(snes_solver_parameters["snes_solver"])
    # solver.solve()
    
    
    # return uh


def getPsi(e, param):
    lamb, mu, alpha = param
    
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    return (0.5*lamb*(tr_e**2 + 0.5*alpha*tr_e**4) +
           mu*(e2 + 0.5*alpha*e2**2))


def getSig(e, param): # it should be defined like that to be able to compute stresses
    lamb, mu, alpha = param

    e = df.variable(e)
    
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    psi = 0.5*lamb*(tr_e**2 + 0.5*alpha*tr_e**4) + mu*(e2 + 0.5*alpha*e2**2)
     
    sig = df.diff(psi,e)
    
    return sig
    
if __name__ == '__main__':
    
    # =========== argument input =================    
    Ny_split =  50
    createMesh = True
    
    # ========== dataset folders ================= 
    # folder = rootDataPath + "/cook"
    folder = './mesh/'
    meshfile = folder + 'mesh_%d.xdmf'%Ny_split
    
    if(createMesh):
        lcar = 44.0/Ny_split  
        gmshMesh = CookMembrane(lcar = lcar)
        gmshMesh.write(savefile = meshfile, opt = 'fenics')
        
    
    metric = {'YOUNG_MODULUS': 100.0,
               'POISSON_RATIO': 0.3,
               'ALPHA': 200.0}
    
    lamb, mu = mech.youngPoisson2lame(metric['POISSON_RATIO'], metric['YOUNG_MODULUS']) 
    
    lamb = df.Constant(lamb)
    mu = df.Constant(mu)
    alpha = df.Constant(metric['ALPHA'])
    
    psi_law = partial(getPsi, param = [lamb, mu, alpha])
    sig_law = partial(getSig, param = [lamb, mu, alpha])

    uh  = solve_cook_multiphenics(meshfile, psi_law, sig_law)

    # sig = getSig(uh, param = [lamb, mu, alpha] )
    
    # sh0 = df.VectorFunctionSpace(uh.function_space().mesh() , 'DG', 0, dim = 3)
    
    # epsh = DDFunction(sh0)
    # sigh = DDFunction(sh0)
    
    # sigh.update(feut.stress2Voigt(sig))
    # epsh.update(symgrad_voigt(uh))
    
    # sigh.rename('sigma', '')
    # uh.rename('u', '')
    
    # iofe.exportXDMF_gen(folder + "cook_vtk.xdmf", fields={'vertex': [uh] , 'cell': [sigh]})
    
    # print(np.min(epsh.data(), axis = 0))
    # print(np.max(epsh.data(), axis = 0))
    
    # data = np.concatenate((epsh.data(), sigh.data()), axis = 1)
    # np.savetxt('database_ref.txt', data, header = '1.0 \n%d 2 3 3'%data.shape[0], comments = '', fmt='%.8e', )




