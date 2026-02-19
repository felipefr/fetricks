"""
This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""




__all__ = ['evaluate_function', 'symgrad', 'BlockSolver', 'CustomLinearSolver'
            'tensor2mandel', 'mandel2tensor', 'tensor4th2mandel', 'tr_mandel', 'Id_mandel_np', 'Id_mandel_df', 'symgrad_mandel',
            'tensor2mandel_np', 'mandel2tensor_np',
            'grad2mandel_vec', 'grad2mandel_ten', 'mandelgrad', 'mandelgrad_ten',
            'Integral',
            'Newton', 'Newton_automatic', 'local_project', 'local_project_given_sol', 'LocalProjector', 
            'Mesh', 'Gmsh',
            'multiscaleMaterialModel', 'multiscaleMaterialModelExpression', 'hyperelasticModel', 'hyperelasticModelExpression',
            'Nonlinear_SNESProblem', 'Nonlinear_SNESSolver',
            'get_Celas_mandel' , 
            'create_piecewise_constant_field',
            'XDMFReader', 'XDMFWriter',
            'getF_fromE', 'getUmandel_fromEmandel']

from .src.plotting.xdmf_io import (XDMFReader, XDMFWriter)

from .src.fenics.multimaterial import create_piecewise_constant_field

from .src.fenics.postprocessing.errors import (error_L2)
from .src.mechanics.truss_utils  import (grad_truss, get_mesh_truss, get_tangent_truss, solve_truss, posproc_truss)

from .src.fenics.la.wrapper_solvers import (CustomLinearSolver, CustomNonlinearProblem, 
                                        CustomLinearSolver, CustomNonlinearSolver,
                                        CustomTangentProblem, BlockSolver, picard,
                                        Nonlinear_SNESProblem, Nonlinear_SNESSolver)
from .src.fenics.la.operations import L2norm, L2norm_given_form
from .src.fenics.mesh.mesh import Mesh
from .src.fenics.mesh.mesh_utils import generate_rectangle_mesh, generate_unit_square_mesh, get_cell_volume

from .src.fenics.bcs_utils import neumannbc, dirichletbc
from .src.fenics.mesh.wrapper_gmsh import gmshio # uses new meshio
from .src.fenics.fem_utils import mixed_functionspace, CustomQuadratureSpace, QuadratureEvaluator
from .src.fenics.math_utils import symgrad, integral, evaluate_function
from .src.plotting.misc import (load_latex_options, set_pallette, plot_mean_std, plot_mean_std_nolegend, plot_fill_std)

from .src.mechanics.elasticity_conversions import get_Celas_mandel
from .src.mechanics.misc import create_piecewise_constant_field



from .src.mechanics.material_models import (psi_ciarlet, psi_ciarlet_C, psi_ciarlet_F, psi_hookean_nonlinear_lame, get_stress_tang_from_psi, 
                                        PK2_ciarlet_C_np, psi_hartmannneff, psi_hartmannneff_C, PK2_hartmannneff_C_np)

from .src.mechanics.hyperlasticity_utils import (GL2CG_np, plane_strain_CG_np, get_invariants_iso_np, 
                                                get_invariants_iso_np, get_GL_mandel, get_deltaGL_mandel,
                                                getF_fromE, getUmandel_fromEmandel)

from .src.fenics.la.conversions import (as_flatten_2x2, as_flatten_3x3, 
                                    as_unflatten_2x2, as_cross_2x2, as_skew_2x2, flatgrad_2x2, flatsymgrad_2x2,
                                    sym_flatten_3x3_np, as_sym_tensor_3x3_np, as_sym_tensor_3x3, ind_sym_tensor_3x3,
                                    sym_flatten_4x4_np, as_sym_tensor_4x4_np, as_sym_tensor_4x4, ind_sym_tensor_4x4,
                                    sym_flatten_9x9_np, as_sym_tensor_9x9_np, as_sym_tensor_9x9, ind_sym_tensor_9x9 )



# Explicit import conversions
from .src.mechanics import conversions as conv2d
from .src.mechanics import conversions3d as conv3d
from .src.mechanics.conversions import tensor2mandel, mandel2tensor, tensor4th2mandel, tr_mandel, Id_mandel_np, Id_mandel_df, symgrad_mandel
from .src.mechanics.conversions import tensor2mandel_np, mandel2tensor_np, tensor4th2mandel_np
from .src.mechanics.conversions import grad2mandel_vec, grad2mandel_ten, mandelgrad, mandelgrad_ten

# lexigraphic
from .src.mechanics.conversions import (Id_lex_df, Id_lex_np, lex2tensor_np, tensor2lex_np, lex2tensor, 
    tensor2lex, tensor4th2lex, tensor4th2lex_np, tr_lex, grad_lex, macro_strain_lex) 

# unsymetric
from .src.mechanics.conversions import (Id_unsym_df, Id_unsym_np, unsym2tensor_np, tensor2unsym_np, unsym2tensor, 
    tensor2unsym, tensor4th2unsym, tensor4th2unsym_np, tr_unsym, grad_unsym, macro_strain_unsym) 




# from .src.fenics.fem.quadrature_function import QuadratureFunction
# from .src.mechanics.material_model_interface import materialModel , materialModelExpression
# from .src.mechanics.isocoric_isotropic_hyperlastic_material import IsochoricIsotropicHyperelasticMaterial

# from .src.mechanics.generic_gausspoint_expression import genericGaussPointExpression
# from .src.mechanics.multiscale_model import multiscaleModel
# from .src.mechanics.multiscale_model_expression import multiscaleModelExpression
# from .src.mechanics.hyperelastic_model import hyperelasticModel, hyperelasticModelExpression
# from .src.mechanics.incompressible_hyperlasticity_utils import Dev, getSiso, getSvol, getDiso, getDvol

# from .src.fenics.la.operations import outer_overline_ufl, outer_underline_ufl, outer_dot_ufl, outer_dot_mandel_ufl

# from .src.fenics.la.wrapper_solvers import (CustomNonlinearSolver, CustomNonlinearProblem)

# from .src.fenics.postprocessing.misc import load_sol, get_errors


# from .src.fenics.misc import create_quadrature_spaces_mechanics, create_DG_spaces_mechanics, symgrad,  setter

# Conversions
# the default is 2d, if you want use explictly ft.conv2d or ft.conv3d, or even rename it with conv = ft.convXd
# from .src.mechanics.conversions2d import *
# from .src.mechanics import conversions as conv2d
# from .src.mechanics import conversions3d as conv3d

# def get_mechanical_notation_conversor(dim_strain = None, gdim = None):
#     if(gdim):
#         return {2: conv2d, 3: conv3d}[gdim]
#     elif(dim_strain):
#         return {3: conv2d, 6: conv3d}[dim_strain]
    


