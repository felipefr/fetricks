#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:48:27 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np

from .generic_gausspoint_expression import genericGaussPointExpression
from fetricks import *

class materialModelExpression:
    def __init__(self, W, dxm):
        self.strain = df.Function(W) 
        self.projector = LocalProjector(W, dxm)
        
        self.stress = genericGaussPointExpression(self.strain, self.stress , (3,))
        self.tangent = genericGaussPointExpression(self.strain, self.tangent , (3,3,))
        
    def stress(self, e, cell = None):
        pass
    
    def tangent(self,e, cell = None):
        pass
    
    def updateStrain(self, e):
        self.projector(e, self.strain)
