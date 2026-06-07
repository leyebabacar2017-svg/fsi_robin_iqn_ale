from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem

import ufl
import numpy as np

# ==========================================================
# Géométrie
# ==========================================================

L = 1.0
H = 0.1

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, H])],
    [100, 10],
    cell_type=mesh.CellType.triangle,
)

# ==========================================================
# Espace EF vectoriel
# ==========================================================

V = fem.functionspace(
    domain,
    ("Lagrange", 1, (2,))
)

# ==========================================================
# Paramètres matériau
# ==========================================================

E = 1.0e6
nu = 0.3

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# ==========================================================
# Cinématique
# ==========================================================

def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return (
        lmbda * ufl.tr(eps(u)) * ufl.Identity(2)
        + 2.0 * mu * eps(u)
    )

# ==========================================================
# Encastrement à gauche
# ==========================================================

fdim = domain.topology.dim - 1

left_facets = mesh.locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[0], 0.0)
)

bc_dofs = fem.locate_dofs_topological(
    V,
    fdim,
    left_facets
)

u_D = np.array(
    (0.0, 0.0),
    dtype=PETSc.ScalarType
)

bc = fem.dirichletbc(
    u_D,
    bc_dofs,
    V
)

# ==========================================================
# Charge volumique
# ==========================================================

f = fem.Constant(
    domain,
    PETSc.ScalarType((0.0, -1000.0))
)

# ==========================================================
# Formulation variationnelle
# ==========================================================

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(
    sigma(u),
    eps(v)
) * ufl.dx

Lform = ufl.dot(
    f,
    v
) * ufl.dx

# ==========================================================
# Résolution
# ==========================================================

problem = LinearProblem(
    a,
    Lform,
    petsc_options_prefix="beam",
    bcs=[bc],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

uh = problem.solve()

# ==========================================================
# Déplacement maximal
# ==========================================================

u_array = uh.x.array.reshape((-1, 2))

umax = np.max(
    np.sqrt(
        u_array[:, 0]**2 +
        u_array[:, 1]**2
    )
)

if MPI.COMM_WORLD.rank == 0:
    print()
    print("====================================")
    print("Elasticity solve completed")
    print("Maximum displacement =", umax)
    print("====================================")

# ==========================================================
# Sauvegarde
# ==========================================================

with io.XDMFFile(
    domain.comm,
    "results/beam_solution.xdmf",
    "w"
) as xdmf:

    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

if MPI.COMM_WORLD.rank == 0:
    print()
    print("Solution written to:")
    print("results/beam_solution.xdmf")

