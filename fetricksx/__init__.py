"""
This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""




__all__ = ['symgrad', 'BlockSolver', 'CustomLinearSolver'
            'tensor2mandel', 'mandel2tensor', 'tensor4th2mandel', 'tr_mandel', 'Id_mandel_np', 'Id_mandel_df', 'symgrad_mandel',
            'tensor2mandel_np', 'mandel2tensor_np',
            'grad2mandel_vec', 'grad2mandel_ten', 'mandelgrad', 'mandelgrad_ten',
            'Integral',
            'Newton', 'Newton_automatic', 'local_project', 'local_project_given_sol', 'LocalProjector', 
            'Mesh', 'Gmsh',
            'multiscaleMaterialModel', 'multiscaleMaterialModelExpression', 'hyperelasticModel', 'hyperelasticModelExpression']


from .fenics.postprocessing.errors import (error_L2)
from .mechanics.truss_utils  import (grad_truss, get_mesh_truss, get_tangent_truss, solve_truss, posproc_truss)

from .fenics.la.wrapper_solvers import (CustomLinearSolver, CustomNonlinearProblem, CustomLinearSolver, CustomNonlinearSolver,
                                        CustomTangentProblem, BlockSolver, picard)
from .fenics.la.operations import L2norm, L2norm_given_form
from .fenics.mesh.mesh import Mesh
from .fenics.mesh.mesh_utils import generate_rectangle_mesh, generate_unit_square_mesh, get_cell_volume

from .fenics.bcs_utils import neumannbc, dirichletbc
from .fenics.mesh.wrapper_gmsh import gmshio # uses new meshio
from .fenics.fem_utils import mixed_functionspace, CustomQuadratureSpace, QuadratureEvaluator
from .fenics.math_utils import symgrad, integral
from .plotting.misc import (load_latex_options, set_pallette, plot_mean_std, plot_mean_std_nolegend, plot_fill_std)

from .mechanics.elasticity_conversions import Celas_mandel
from .mechanics.misc import create_piecewise_constant_field



# from .fenics.fem.quadrature_function import QuadratureFunction
# from .mechanics.material_model_interface import materialModel , materialModelExpression
# from .mechanics.isocoric_isotropic_hyperlastic_material import IsochoricIsotropicHyperelasticMaterial
from .mechanics.material_models import (psi_ciarlet, psi_ciarlet_C, psi_ciarlet_F, psi_hookean_nonlinear_lame, get_stress_tang_from_psi, 
                                        PK2_ciarlet_C_np, psi_hartmannneff, psi_hartmannneff_C, PK2_hartmannneff_C_np)
# from .mechanics.generic_gausspoint_expression import genericGaussPointExpression
# from .mechanics.multiscale_model import multiscaleModel
# from .mechanics.multiscale_model_expression import multiscaleModelExpression
# from .mechanics.hyperelastic_model import hyperelasticModel, hyperelasticModelExpression
# from .mechanics.incompressible_hyperlasticity_utils import Dev, getSiso, getSvol, getDiso, getDvol
from .mechanics.hyperlasticity_utils import GL2CG_np, plane_strain_CG_np, get_invariants_iso_np, get_invariants_iso_np, get_GL_mandel, get_deltaGL_mandel

from .fenics.la.conversions import (as_flatten_2x2, as_flatten_3x3, 
                                    as_unflatten_2x2, as_cross_2x2, as_skew_2x2, flatgrad_2x2, flatsymgrad_2x2,
                                    sym_flatten_3x3_np, as_sym_tensor_3x3_np, ind_sym_tensor_3x3, as_sym_tensor_3x3)


# from .fenics.la.operations import outer_overline_ufl, outer_underline_ufl, outer_dot_ufl, outer_dot_mandel_ufl

# from .fenics.la.wrapper_solvers import (CustomNonlinearSolver, CustomNonlinearProblem)

# from .fenics.postprocessing.misc import load_sol, get_errors


# from .fenics.misc import create_quadrature_spaces_mechanics, create_DG_spaces_mechanics, symgrad,  setter

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
    


# Explicit import conversions
from .mechanics import conversions as conv2d
from .mechanics import conversions3d as conv3d
from .mechanics.conversions import tensor2mandel, mandel2tensor, tensor4th2mandel, tr_mandel, Id_mandel_np, Id_mandel_df, symgrad_mandel
from .mechanics.conversions import tensor2mandel_np, mandel2tensor_np, tensor4th2mandel_np
from .mechanics.conversions import grad2mandel_vec, grad2mandel_ten, mandelgrad, mandelgrad_ten





