from mpi4py import MPI
import numpy as np
import ufl
import sys

sys.path.append("..")

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem

from common.tags import Tags

from common.materials import (
    lame_parameters
)

from common.elasticity import (
    eps,
    sigma
)

from common.mesh_utils import (
    load_turek_mesh,
    extract_solid_mesh
)

from common.boundary_transfer import (
    transfer_facet_tags_to_submesh
)

from common.boundary_conditions import (
    create_clamp_bc
)

from common.traction import (
    create_interface_boundary,
    create_vertical_traction
)

from common.io_utils import (
    print_mesh_info
)

from common.parameters import (
    TurekParameters
)

# =====================================================
# Lecture du maillage
# =====================================================

domain, cell_tags, facet_tags = \
    load_turek_mesh()

solid_mesh, cell_map, _, _ = \
    extract_solid_mesh(
        domain,
        cell_tags
    )

print_mesh_info(
    solid_mesh,
    "Solid mesh"
)

# =====================================================
# Mesures
# =====================================================

dx = ufl.dx(domain=solid_mesh)

# =====================================================
# Espace EF
# =====================================================

V = fem.functionspace(
    solid_mesh,
    ("Lagrange", 1, (2,))
)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# =====================================================
# Paramètres matériau
# =====================================================

E = TurekParameters.E

nu = TurekParameters.nu

mu, lmbda = lame_parameters(
    E,
    nu
)

# =====================================================
# Encastrement x = 0.25
# =====================================================

solid_facet_tags = \
    transfer_facet_tags_to_submesh(
        domain,
        facet_tags,
        solid_mesh,
        cell_map
    )

print()
print("Tags présents sur le sous-maillage solide :")
print(np.unique(solid_facet_tags.values))

print(
    "Interface facets =",
    len(
        solid_facet_tags.find(
            Tags.INTERFACE
        )
    )
)

bc, clamp_facets = \
    create_clamp_bc(
        solid_mesh,
        solid_facet_tags,
        V
    )
    
print()
print("Clamp facets =", len(clamp_facets))

# =====================================================
# Traction
# =====================================================

TRACTION = TurekParameters.traction

traction = create_vertical_traction(
    solid_mesh,
    TRACTION
)

print()
print("Applied traction =", TRACTION)

a = (
    ufl.inner(
        sigma(u, mu, lmbda),
        eps(v)
    )
    * dx
)

# =====================================================
# Traction sur l'interface fluide-structure
# =====================================================

interface_facets, ds_interface = \
    create_interface_boundary(
        solid_mesh,
        solid_facet_tags
    )

print(
    "Interface facets =",
    len(interface_facets)
)

L = (
    ufl.dot(
        traction,
        v
    )
    * ds_interface(1)
)

# =====================================================
# Résolution
# =====================================================

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="turek_linear",
    petsc_options={
        "ksp_type":"preonly",
        "pc_type":"lu"
    }
)

uh = problem.solve()

# =====================================================
# Von Mises
# =====================================================

s = sigma(
    uh,
    mu,
    lmbda
)

von_mises_expr = ufl.sqrt(
    s[0,0]**2
    - s[0,0]*s[1,1]
    + s[1,1]**2
    + 3.0*s[0,1]**2
)

Wvm = fem.functionspace(
    solid_mesh,
    ("DG", 0)
)

von_mises = fem.Function(Wvm)

expr_vm = fem.Expression(
    von_mises_expr,
    Wvm.element.interpolation_points
)

von_mises.interpolate(expr_vm)

von_mises.name = "von_mises"
# =====================================================
# Post-traitement
# =====================================================

u_array = uh.x.array.reshape((-1,2))

umax = np.max(
    np.sqrt(
        u_array[:,0]**2 +
        u_array[:,1]**2
    )
)

if MPI.COMM_WORLD.rank == 0:

    print()
    print("====================================")
    print("Linear elasticity solved")
    print("Maximum displacement =",umax)
    print("====================================")
    
with io.XDMFFile(
    solid_mesh.comm,
    "results/linear_turek.xdmf",
    "w"
) as xdmf:

    xdmf.write_mesh(solid_mesh)
    xdmf.write_function(uh)
    xdmf.write_function(von_mises)
    
print()
print("Solution saved:")
print("results/linear_turek.xdmf")

