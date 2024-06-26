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

def load_latex_options():
    plt.rc("text", usetex = True)
    plt.rc("font", family = 'serif')
    plt.rc("font", size = 12)
    # plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts}')
    # plt.rcParams["mathtext.fontset"] = "cm"

def set_pallette(palette = None):
    palletteCounter = 0
    pallette = palette if palette else ['blue','red','green']

def plot_mean_std(x, y, l='', linetypes = ['-o','--','--'], axis = 0):
    plt.plot(x, np.mean(y, axis = axis), linetypes[0], label = l)
    plt.plot(x, np.mean(y, axis = axis) + np.std(y, axis = axis) , linetypes[1], label = l + ' + std')
    plt.plot(x, np.mean(y, axis = axis) - np.std(y, axis = axis) , linetypes[2], label = l + ' - std')
    
    
    
def plot_mean_std_nolegend(x, y, l='', linetypes = ['-o','--','--'], axis = 0):
    plt.plot(x, np.mean(y, axis = axis), linetypes[0], label = l)
    plt.plot(x, np.mean(y, axis = axis) + np.std(y, axis = axis) , linetypes[1])
    plt.plot(x, np.mean(y, axis = axis) - np.std(y, axis = axis) , linetypes[2])
    
def plot_fill_std(x, y, l='', linetypes = ['-o','--','--'], axis = 0):
    global palletteCounter, pallette
    
    elementWiseMax = lambda a,b : np.array([max(ai,b) for ai in a])
    tol = 1.0e-6
    mean = np.mean(y, axis = axis) 
    std = np.std(y, axis = axis)
    
    
    color = pallette[palletteCounter]
    plt.plot(x, mean, linetypes[0], label = l, color = color)
    plt.fill_between( x,  elementWiseMax(mean - std, tol) , 
                           mean + std, facecolor = color , alpha = 0.3)

    palletteCounter += 1