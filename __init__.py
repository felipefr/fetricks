from fetricks.mechanics.conversions import stress2voigt, strain2voigt, voigt2strain, voigt2stress, mandel2voigtStrain, mandel2voigtStress
from fetricks.mechanics.conversions import tensor2mandel, mandel2tensor, tensor4th2mandel, tr_mandel, Id_mandel_np, Id_mandel_df, symgrad_mandel, symgrad_voigt
from fetricks.mechanics.conversions import tensor2mandel_np, mandel2tensor_np
from fetricks.mechanics.conversions import grad2mandel_vec, grad2mandel_ten, mandelgrad, mandelgrad_ten

from fetricks.fenics.misc import symgrad, Integral
from fetricks.fenics.la.wrapper_solvers import Newton, Newton_automatic, local_project, local_project_given_sol, LocalProjector
from fetricks.fenics.mesh.mesh import Mesh

# import pygmsh
# if('built_in' in pygmsh.__all__):
#     from fetricks.fenics.mesh.wrapper_gmsh_legacy_pygmsh import GmshIO # uses pygmsh 6.0.2 and meshio 3.3.1
# else: 
#     from fetricks.fenics.mesh.wrapper_gmsh import GmshIO # uses new meshio
    
# force to use new version
from fetricks.fenics.mesh.wrapper_gmsh import GmshIO # uses new meshio
    
from fetricks.fenics.material.multiscale_model import multiscaleModel, multiscaleModelExpression
from fetricks.fenics.material.hyperelastic_model import hyperelasticModel, hyperelasticModelExpression


__all__ = ['stress2voigt', 'strain2voigt', 'voigt2strain', 'voigt2stress', 'mandel2voigtStrain', 'mandel2voigtStress',
'tensor2mandel', 'mandel2tensor', 'tensor4th2mandel', 'tr_mandel', 'Id_mandel_np', 'Id_mandel_df', 'symgrad_mandel', 'symgrad_voigt',
'tensor2mandel_np', 'mandel2tensor_np',
'grad2mandel_vec', 'grad2mandel_ten', 'mandelgrad', 'mandelgrad_ten',
'symgrad', 'Integral',
'Newton', 'Newton_automatic', 'local_project', 'local_project_given_sol', 'LocalProjector', 
'Mesh', 'Gmsh',
'multiscaleMaterialModel', 'multiscaleMaterialModelExpression', 'hyperelasticModel', 'hyperelasticModelExpression']



from .fenics.la.conversions import (as_flatten_2x2, as_flatten_3x3, 
                                    as_unflatten_2x2, as_cross_2x2, as_skew_2x2)


from .fenics.la.wrapper_solvers import (CustomNonlinearSolver, CustomNonlinearProblem)
from .mechanics.material_models import (psi_ciarlet, get_stress_tang_from_psi)