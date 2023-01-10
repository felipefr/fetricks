from fetricks.mechanics.conversions import stress2voigt, strain2voigt, voigt2strain, voigt2stress, mandel2voigtStrain, mandel2voigtStress
from fetricks.mechanics.conversions import tensor2mandel, mandel2tensor, tensor4th2mandel, tr_mandel, Id_mandel_np, Id_mandel_df, symgrad_mandel, symgrad_voigt
from fetricks.mechanics.conversions import tensor2mandel_np, mandel2tensor_np
from fetricks.mechanics.conversions import grad2mandel_vec, grad2mandel_ten, mandelgrad, mandelgrad_ten

from fetricks.fenics.misc import symgrad, Integral
from fetricks.fenics.la.wrapper_solvers import Newton, Newton_automatic, local_project, local_project_given_sol, LocalProjector
from fetricks.fenics.mesh.mesh import Mesh

import pygmsh
if('built_in' in pygmsh.__all__): #  available in the version 6.0.2
    from fetricks.fenics.mesh.wrapper_gmsh import Gmsh
else: 
    from fetricks.fenics.mesh.wrapper_gmsh_new import Gmsh # new version pygmsh (limited functionality)
    
    
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
