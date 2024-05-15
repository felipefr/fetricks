#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:39:16 2024

@author: felipe
"""

import dolfin as df

def MixedFiniteElementSpace(mesh, family, degree):
    FEs = [df.FiniteElement(f, mesh.ufl_cell(), k) for f, k in zip(family, degree) ]
    We = df.MixedElement(FEs)
    Wh = df.FunctionSpace(mesh, We)
    return Wh