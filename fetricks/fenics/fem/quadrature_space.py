#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:48:13 2025

@author: felipe
"""

"""
Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@u-pec.fr>, or
<f.rocha.felipe@gmail.com>

"""

import dolfin as df
# CRITICAL = 50 , ERROR = 40 , WARNING = 30, INFO = 20, PROGRESS = 16, TRACE = 13, DBG = 10
df.set_log_level(50)


class QuadratureSpace(df.FunctionSpace):
    
    def __init__(self, mesh, dim, degree_quad, representation = 'Quadrature'): # degree_quad seems a keyword

        self.representation = representation
        self.degree_quad_ = degree_quad
        self.mesh = mesh
        
        ufl_cell = mesh.ufl_cell()
        
        if(representation == 'Quadrature'):
            import warnings
            from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
            warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
        
            df.parameters["form_compiler"]["representation"] = 'quadrature'
            
            self.Qe = df.VectorElement("Quadrature", ufl_cell, 
                                       degree = self.degree_quad_, dim = dim, quad_scheme='default')
            
            
            self.metadata = {"quadrature_degree": self.degree_quad_, "quadrature_scheme": "default"}
            self.dxm = df.Measure( 'dx', mesh, metadata= self.metadata)
            
        elif(representation == "DG"):
            self.Qe = df.VectorElement("DG", ufl_cell, degree = self.degree_quad_ - 1, dim = dim)
            self.dxm = df.Measure('dx', mesh)
            

        super().__init__(mesh, self.Qe ) # for stress
    
    # scalar space of the present one (necessary to the distance to make sure dist is evaluated on the right spot)
    def get_scalar_space(self):
        if(self.representation == "Quadrature"):
            Qe = df.VectorElement("Quadrature", self.mesh.ufl_cell(), 
                                  degree = self.sub(0).ufl_element().degree(), dim = 1, quad_scheme='default')
            sh0 = df.FunctionSpace(self.mesh(), Qe ) # for stress
        
        elif(self.representation == "DG"):
            sh0 = df.FunctionSpace(self.mesh() , 'DG', self.sub(0).ufl_element().degree())    
            
        return sh0
