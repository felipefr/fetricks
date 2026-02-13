"""

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df


# (key, space, label) (label stands for the name of the function in the file)
def load_sol(filename, keys, spaces, labels = None ):
    
    if(not labels): labels = keys
    
    sol = {key: df.Function(Uh) for key, Uh, label in zip(keys, spaces, labels)} 

    with df.XDMFFile(filename) as f:
        for key, Uh, label in zip(keys, spaces, labels):
            f.read_checkpoint(sol[key], label)
            
    return sol

def get_errors(sol, sol_ref, keys, norm = None, keys_ref = None):
    if(not keys_ref): keys_ref = keys
    errors_abs = { key : norm(sol[key] - sol_ref[key_ref]) for key, key_ref in zip(keys, keys_ref)}
    errors_rel = { key : errors_abs[key]/norm(sol_ref[key_ref]) for key, key_ref in zip(keys, keys_ref)}
    return errors_abs, errors_rel

def visualiseStresses(test, pred = None, figNum = 1, savefig = None):
    n = test.shape[1]
    indx =  np.arange(1,n,3)
    indy =  np.arange(2,n,3)

    plt.figure(figNum,(8,8))
    for i in range(test.shape[0]):
        plt.scatter(test[i,indx], test[i,indy], marker = 'o',  linewidth = 5)

        if(type(pred) != type(None)):
            plt.scatter(pred[i,indx], pred[i,indy], marker = '+', linewidth = 5)


    plt.legend(['test', 'pred'])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    
    if(type(savefig) != type(None)):
        plt.savefig(savefig)
 
    
def visualiseStresses9x9(test, pred = None, figNum = 1, savefig = None):
    n = test.shape[1]
    indx =  np.arange(1,n,3)
    indy =  np.arange(2,n,3)

    plt.figure(figNum,(13,12))
    for i in range(test.shape[0]):
        plt.subplot('33' + str(i+1))
        
        plt.scatter(test[i,indx], test[i,indy], marker = 'o',  linewidth = 5)
        if(type(pred) != type(None)):
            plt.scatter(pred[i,indx], pred[i,indy], marker = '+', linewidth = 5)

        plt.legend(['test', 'pred'])
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.grid()

    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    
    if(type(savefig) != type(None)):
        plt.savefig(savefig)
 
    
    
 
def visualiseScatterErrors(test, pred, labels, gamma = 0.0, figNum = 1, savefig = None):
        
    plt.figure(figNum,(13,12))
    n = test.shape[1]
    
    for i in range(n): 
        plt.subplot('33' + str(i+1))
        plt.scatter(pred[:,i],test[:,i], marker = '+', linewidths = 0.1)
        xy = np.linspace(np.min(test[:,i]),np.max(test[:,i]),2)
        plt.plot(xy,xy,'-',color = 'black')
        plt.xlabel('test ' + labels[i])
        plt.ylabel('prediction ' + labels[i])
        plt.grid()
        
    for i in range(n): 
        plt.subplot('33' + str(i+4))
        plt.scatter(test[:,i],test[:,i] - pred[:,i], marker = '+', linewidths = 0.1)
        plt.xlabel('test ' + labels[i])
        plt.ylabel('error (test - pred) ' + labels[i])
        plt.grid()
    
    for i in range(n): 
        plt.subplot('33' + str(i+7))
        plt.scatter(test[:,i],(test[:,i] - pred[:,i])/(np.abs(test[:,i]) + gamma), marker = '+', linewidths = 0.1)
        plt.xlabel('test ' + labels[i])
        plt.ylabel('error rel (test - pred)/(test + gamma) ' + labels[i])
        plt.grid()
    
    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    
    if(type(savefig) != type(None)):
        plt.savefig(savefig)


