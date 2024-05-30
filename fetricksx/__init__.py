"""
This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

from .fenics.postprocessing.errors import (error_L2)
from .mechanics.truss_utils  import (grad_truss, get_mesh_truss, get_tangent_truss, solve_truss, posproc_truss)

from .fenics.la.wrapper_solvers import BlockSolver
from .fenics.mesh.mesh import Mesh

# import pygmsh
# if('built_in' in pygmsh.__all__):
#     from fetricks.fenics.mesh.wrapper_gmsh_legacy_pygmsh import GmshIO # uses pygmsh 6.0.2 and meshio 3.3.1
# else: 
#     from fetricks.fenics.mesh.wrapper_gmsh import GmshIO # uses new meshio
    
# force to use new version
from .fenics.mesh.wrapper_gmsh import GmshIO # uses new meshio


# from .fenics.fem.quadrature_function import QuadratureFunction


# from .mechanics.material_model_interface import materialModel , materialModelExpression
# from .mechanics.isocoric_isotropic_hyperlastic_material import IsochoricIsotropicHyperelasticMaterial
from .mechanics.material_models import (psi_ciarlet, psi_ciarlet_C, psi_hookean_nonlinear_lame, get_stress_tang_from_psi, 
                                        PK2_ciarlet_C_np, psi_hartmannneff, psi_hartmannneff_C, PK2_hartmannneff_C_np)
# from .mechanics.generic_gausspoint_expression import genericGaussPointExpression
# from .mechanics.multiscale_model import multiscaleModel
# from .mechanics.multiscale_model_expression import multiscaleModelExpression
# from .mechanics.hyperelastic_model import hyperelasticModel, hyperelasticModelExpression
# from .mechanics.incompressible_hyperlasticity_utils import Dev, getSiso, getSvol, getDiso, getDvol
from .mechanics.hyperlasticity_utils import GL2CG_np, plane_strain_CG_np, get_invariants_iso_np, get_invariants_iso_np, get_GL_mandel, get_deltaGL_mandel

# from .fenics.la.conversions import (as_flatten_2x2, as_flatten_3x3, 
                                    # as_unflatten_2x2, as_cross_2x2, as_skew_2x2, flatgrad_2x2, flatsymgrad_2x2,
                                    # sym_flatten_3x3_np, as_sym_tensor_3x3_np, ind_sym_tensor_3x3, as_sym_tensor_3x3)


# from .fenics.la.operations import outer_overline_ufl, outer_underline_ufl, outer_dot_ufl, outer_dot_mandel_ufl

# from .fenics.la.wrapper_solvers import (CustomNonlinearSolver, CustomNonlinearProblem)

# from .fenics.postprocessing.misc import load_sol, get_errors

# from .fenics.bcs.neumann import (NeumannTensorSource, NeumannVectorSource, NeumannBC, NeumannVectorBC, NeumannVectorBC_given_normal,
#                                NeumannTensorSourceCpp, NeumannVectorSourceCpp)
# from .fenics.misc import create_quadrature_spaces_mechanics, create_DG_spaces_mechanics, symgrad, Integral, setter

# Conversions
# the default is 2d, if you want use explictly ft.conv2d or ft.conv3d, or even rename it with conv = ft.convXd
# from .mechanics.conversions2d import *
# from .mechanics import conversions as conv2d
# from .mechanics import conversions3d as conv3d

# def get_mechanical_notation_conversor(dim_strain = None, gdim = None):
#     if(gdim):
#         return {2: conv2d, 3: conv3d}[gdim]
#     elif(dim_strain):
#         return {3: conv2d, 6: conv3d}[dim_strain]
    


# Explicit import conversionss
from .mechanics.conversions import stress2voigt, strain2voigt, voigt2strain, voigt2stress, mandel2voigtStrain, mandel2voigtStress
from .mechanics.conversions import tensor2mandel, mandel2tensor, tensor4th2mandel, tr_mandel, Id_mandel_np, Id_mandel_df, symgrad_mandel, symgrad_voigt
from .mechanics.conversions import tensor2mandel_np, mandel2tensor_np, tensor4th2mandel_np
from .mechanics.conversions import grad2mandel_vec, grad2mandel_ten, mandelgrad, mandelgrad_ten




from .plotting.misc import *

__all__ = ['BlockSolver', 
           'stress2voigt', 'strain2voigt', 'voigt2strain', 'voigt2stress', 'mandel2voigtStrain', 'mandel2voigtStress',
            'tensor2mandel', 'mandel2tensor', 'tensor4th2mandel', 'tr_mandel', 'Id_mandel_np', 'Id_mandel_df', 'symgrad_mandel', 'symgrad_voigt',
            'tensor2mandel_np', 'mandel2tensor_np',
            'grad2mandel_vec', 'grad2mandel_ten', 'mandelgrad', 'mandelgrad_ten',
            'Integral',
            'Newton', 'Newton_automatic', 'local_project', 'local_project_given_sol', 'LocalProjector', 
            'Mesh', 'Gmsh',
            'multiscaleMaterialModel', 'multiscaleMaterialModelExpression', 'hyperelasticModel', 'hyperelasticModelExpression']
