#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:10:31 2024

@author: felipe
"""

import gmsh

# n : number of divisions per edge
def generate_msh_unit_square_mesh(n, msh_file): 
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
    geom = gmsh.model.geo
    
    p = []
    p.append(geom.add_point(0.0, 0.0, 0.0))
    p.append(geom.add_point(1.0, 0.0, 0.0))
    p.append(geom.add_point(1.0, 1.0, 0.0))
    p.append(geom.add_point(0.0, 1.0, 0.0))

        
    l = []
    l.append(geom.add_line(p[0], p[1]))
    l.append(geom.add_line(p[1], p[2]))
    l.append(geom.add_line(p[2], p[3]))
    l.append(geom.add_line(p[3], p[0]))
    
    ll = [geom.add_curve_loop([l[0], l[1], l[2], l[3]])]
    s = [geom.add_plane_surface(ll)]
    
    geom.synchronize()
    
    for li in l:
        gmsh.model.mesh.set_transfinite_curve(li, n+1)
    
    gmsh.model.mesh.set_transfinite_surface(s[0],arrangement="Left")
    
    gmsh.model.add_physical_group(2, s, 0)
    for i in range(4):
        gmsh.model.add_physical_group(1, [l[i]], i+1) # topology, list objects, flag
        
    gmsh.model.mesh.generate(dim=2)
    gmsh.write(msh_file)
    
#    domain, markers, facets = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    
    gmsh.finalize()
    
    return