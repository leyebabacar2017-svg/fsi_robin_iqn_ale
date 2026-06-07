from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

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
    [80, 8],
    cell_type=mesh.CellType.triangle
)

# ==========================================================
# Espace EF
# ==========================================================

V = fem.functionspace(domain, ("Lagrange", 2, (2,)))

u = fem.Function(V)
u.name = "displacement"

v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

# ==========================================================
# Paramètres matériau
# ==========================================================

E = 1.0e6
nu_mat = 0.3

mu = E / (2.0 * (1.0 + nu_mat))
lmbda = E * nu_mat / ((1.0 + nu_mat) * (1.0 - 2.0 * nu_mat))

# ==========================================================
# Cinématique
# ==========================================================

d = domain.geometry.dim

I = ufl.Identity(d)

F = I + ufl.grad(u)

C = F.T * F

Ic = ufl.tr(C)

J = ufl.det(F)

# ==========================================================
# Énergie Neo-Hookean
# ==========================================================

psi = (
    (mu / 2.0) * (Ic - d)
    - mu * ufl.ln(J)
    + (lmbda / 2.0) * (ufl.ln(J))**2
)

# ==========================================================
# Traction appliquée à droite
# ==========================================================

T = fem.Constant(
    domain,
    PETSc.ScalarType((0.0, -100.0))
)

# ==========================================================
# Frontière droite
# ==========================================================

fdim = domain.topology.dim - 1

right_facets = mesh.locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[0], L)
)

facet_indices = np.array(right_facets, dtype=np.int32)
facet_values = np.ones_like(facet_indices)

facet_tag = mesh.meshtags(
    domain,
    fdim,
    facet_indices,
    facet_values
)

ds = ufl.Measure(
    "ds",
    domain=domain,
    subdomain_data=facet_tag
)

# ==========================================================
# Encastrement à gauche
# ==========================================================

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

u_bc = np.array(
    (0.0, 0.0),
    dtype=PETSc.ScalarType
)

bc = fem.dirichletbc(
    u_bc,
    bc_dofs,
    V
)

# ==========================================================
# Potentiel total
# ==========================================================

Pi = psi * ufl.dx - ufl.dot(T, u) * ds(1)

# ==========================================================
# Résidu et Jacobienne
# ==========================================================

R = ufl.derivative(Pi, u, v)

J_form = ufl.derivative(R, u, du)

# ==========================================================
# Problème non linéaire
# ==========================================================

problem = NewtonSolverNonlinearProblem(
    R,
    u,
    bcs=[bc],
    J=J_form
)

solver = NewtonSolver(
    MPI.COMM_WORLD,
    problem
)

solver.atol = 1e-8
solver.rtol = 1e-8

solver.max_it = 50

solver.report = True

solver.error_on_nonconvergence = False

solver.relaxation_parameter = 0.5

solver.convergence_criterion = "incremental"

# ==========================================================
# Résolution
# ==========================================================

n, converged = solver.solve(u)

u.x.scatter_forward()

if MPI.COMM_WORLD.rank == 0:

    print()
    print("===================================")
    print("Neo-Hookean solve completed")
    print("Iterations :", n)
    print("Converged  :", converged)
    print("===================================")

    if not converged:
        print("WARNING : Newton did not fully converge")

# ==========================================================
# Déplacement maximal
# ==========================================================

u_array = u.x.array.real.reshape((-1, 2))

umax = np.max(
    np.sqrt(
        u_array[:, 0]**2 +
        u_array[:, 1]**2
    )
)

if MPI.COMM_WORLD.rank == 0:
    print("Maximum displacement =", umax)

# ==========================================================
# Sauvegarde
# ==========================================================

# ==========================================================
# Interpolation P2 -> P1 pour visualisation
# ==========================================================

Vout = fem.functionspace(
    domain,
    ("Lagrange", 1, (2,))
)

u_out = fem.Function(Vout)
u_out.interpolate(u)

with io.XDMFFile(
    domain.comm,
    "results/neo_hookean_solution.xdmf",
    "w"
) as xdmf:

    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)

if MPI.COMM_WORLD.rank == 0:
    print()
    print("Solution written to:")
    print("results/neo_hookean_solution.xdmf")
