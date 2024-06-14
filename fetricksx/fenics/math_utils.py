#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:08:19 2024

@author: felipe
"""
import ufl

# NON-FLATTENED FUNCTIONS
def symgrad(v): 
    return ufl.sym(ufl.grad(v))
