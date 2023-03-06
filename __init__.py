"""
This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

from .fenics.la.wrapper_solvers import Newton, Newton_automatic, local_project, local_project_given_sol, LocalProjector
from .fenics.mesh.mesh import Mesh

# import pygmsh
# if('built_in' in pygmsh.__all__):
#     from fetricks.fenics.mesh.wrapper_gmsh_legacy_pygmsh import GmshIO # uses pygmsh 6.0.2 and meshio 3.3.1
# else: 
#     from fetricks.fenics.mesh.wrapper_gmsh import GmshIO # uses new meshio
    
# force to use new version
from .fenics.mesh.wrapper_gmsh import GmshIO # uses new meshio
    
from .mechanics.multiscale_model import multiscaleModel
from .mechanics.multiscale_model_expression import multiscaleModelExpression
from .mechanics.hyperelastic_model import hyperelasticModel, hyperelasticModelExpression

from .fenics.misc import create_quadrature_spaces_mechanics, symgrad, Integral, setter
from .fenics.la.conversions import (as_flatten_2x2, as_flatten_3x3, 
                                    as_unflatten_2x2, as_cross_2x2, as_skew_2x2, flatgrad_2x2, flatsymgrad_2x2,
                                    sym_flatten_3x3_np, as_sym_tensor_3x3_np, ind_sym_tensor_3x3, as_sym_tensor_3x3)

from .fenics.la.wrapper_solvers import (CustomNonlinearSolver, CustomNonlinearProblem)
from .mechanics.material_models import (psi_ciarlet, psi_hookean_nonlinear_lame, get_stress_tang_from_psi)

from .fenics.bcs.neumann import NeumannTensorSource, NeumannVectorSource, NeumannBC, NeumannVectorBC, NeumannVectorBC_given_normal


from .mechanics.conversions import stress2voigt, strain2voigt, voigt2strain, voigt2stress, mandel2voigtStrain, mandel2voigtStress
from .mechanics.conversions import tensor2mandel, mandel2tensor, tensor4th2mandel, tr_mandel, Id_mandel_np, Id_mandel_df, symgrad_mandel, symgrad_voigt
from .mechanics.conversions import tensor2mandel_np, mandel2tensor_np
from .mechanics.conversions import grad2mandel_vec, grad2mandel_ten, mandelgrad, mandelgrad_ten

# from .mechanics.conversions3d import stress2voigt, strain2voigt, voigt2strain, voigt2stress, mandel2voigtStrain, mandel2voigtStress
# from .mechanics.conversions3d import tensor2mandel, mandel2tensor, tensor4th2mandel, tr_mandel, Id_mandel_np, Id_mandel_df, symgrad_mandel, symgrad_voigt
# from .mechanics.conversions3d import tensor2mandel_np, mandel2tensor_np
# from .mechanics.conversions3d import grad2mandel_vec, grad2mandel_ten, mandelgrad, mandelgrad_ten



# __all__ = ['stress2voigt', 'strain2voigt', 'voigt2strain', 'voigt2stress', 'mandel2voigtStrain', 'mandel2voigtStress',
# 'tensor2mandel', 'mandel2tensor', 'tensor4th2mandel', 'tr_mandel', 'Id_mandel_np', 'Id_mandel_df', 'symgrad_mandel', 'symgrad_voigt',
# 'tensor2mandel_np', 'mandel2tensor_np',
# 'grad2mandel_vec', 'grad2mandel_ten', 'mandelgrad', 'mandelgrad_ten',
# 'symgrad', 'Integral',
# 'Newton', 'Newton_automatic', 'local_project', 'local_project_given_sol', 'LocalProjector', 
# 'Mesh', 'Gmsh',
# 'multiscaleMaterialModel', 'multiscaleMaterialModelExpression', 'hyperelasticModel', 'hyperelasticModelExpression']
