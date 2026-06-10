from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix

from slepc4py import SLEPc

import ufl

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

from common.materials import (
    lame_parameters
)

from common.elasticity import (
    eps,
    sigma
)

from common.io_utils import (
    print_mesh_info
)

from common.parameters import (
    TurekParameters
)

# =====================================================
# Mesh
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
    "Modal analysis"
)

# =====================================================
# FE space
# =====================================================

V = fem.functionspace(
    solid_mesh,
    ("Lagrange", 1, (2,))
)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# =====================================================
# Material
# =====================================================

rho = TurekParameters.rho

E = TurekParameters.E
nu = TurekParameters.nu

mu, lmbda = lame_parameters(
    E,
    nu
)

# =====================================================
# Boundary conditions
# =====================================================

solid_facet_tags = \
    transfer_facet_tags_to_submesh(
        domain,
        facet_tags,
        solid_mesh,
        cell_map
    )

bc, clamp_facets = \
    create_clamp_bc(
        solid_mesh,
        solid_facet_tags,
        V
    )

blocked_dofs = bc.dof_indices()[0]

print()
print(
    "Blocked dofs =",
    len(blocked_dofs)
)

print(
    "Global nodes =",
    solid_mesh.geometry.x.shape[0]
)

print(
    "Blocked dofs =",
    len(
        bc.dof_indices()[0]
    )
)

print()
print(
    "Min blocked dof =",
    np.min(blocked_dofs)
)

print(
    "Max blocked dof =",
    np.max(blocked_dofs)
)

print()
print(
    "Clamp facets =",
    len(clamp_facets)
)

# =====================================================
# Matrices
# =====================================================

dx = ufl.dx(domain=solid_mesh)

m_form = rho * ufl.inner(
    u,
    v
) * dx

k_form = ufl.inner(
    sigma(
        u,
        mu,
        lmbda
    ),
    eps(v)
) * dx

M = assemble_matrix(
    fem.form(m_form),
    bcs=[bc]
)
M.assemble()

K = assemble_matrix(
    fem.form(k_form),
    bcs=[bc]
)
K.assemble()

print()
print("Matrices assembled")

print()
print("||M||F =", M.norm())
print("||K||F =", K.norm())

ndofs = K.getSize()[0]

all_dofs = np.arange(
    ndofs,
    dtype=np.int32
)

free_dofs = np.setdiff1d(
    all_dofs,
    blocked_dofs
)

print()
print(
    "Free dofs =",
    len(free_dofs)
)

print(
    "Global dofs =",
    K.getSize()[0]
)

is_free = PETSc.IS().createGeneral(
    free_dofs
)

Kff = K.createSubMatrix(
    is_free,
    is_free
)

Mff = M.createSubMatrix(
    is_free,
    is_free
)

# =====================================================
# Eigenvalue solver
# =====================================================

eps_solver = SLEPc.EPS().create(
    MPI.COMM_WORLD
)

eps_solver.setType(
    SLEPc.EPS.Type.KRYLOVSCHUR
)

eps_solver.setOperators(
    Kff,
    Mff
)

eps_solver.setProblemType(
    SLEPc.EPS.ProblemType.GHEP
)

eps_solver.setDimensions(
    nev=20
)

eps_solver.setTarget(
    0.0
)

eps_solver.setWhichEigenpairs(
    SLEPc.EPS.Which.TARGET_MAGNITUDE
)

st = eps_solver.getST()

st.setType(
    SLEPc.ST.Type.SINVERT
)

eps_solver.solve()

print()
print(
    "Solver type =",
    eps_solver.getType()
)

print()
print(
    "Solver type =",
    eps_solver.getType()
)

print(
    "Problem type = GHEP"
)

eps_solver.solve()

# =====================================================
# Frequencies
# =====================================================

nconv = eps_solver.getConverged()

print()
print(
    "Converged modes =",
    nconv
)

for i in range(nconv):

    eig = eps_solver.getEigenvalue(i)

    omega = np.sqrt(eig.real)

    freq = omega/(2*np.pi)

    print(
        f"Mode {i+1}: "
        f"{freq:.6f} Hz"
    )

# =====================================================
# Export first mode
# =====================================================

#if nconv > 0:

#    vr, vi = K.getVecs()

#    eps_solver.getEigenvector(
#        0,
#        vr,
#        vi
#    )

#    mode1 = fem.Function(V)

#    mode1.x.array[:] = vr.array

#    mode1.name = "mode_1"

#    with io.XDMFFile(
#        MPI.COMM_WORLD,
#        "results/mode1.xdmf",
#        "w"
#    ) as xdmf:

#        xdmf.write_mesh(
#            solid_mesh
#        )

#        xdmf.write_function(
#            mode1
#        )

#    print()
#    print(
#        "Mode 1 saved:"
#    )
#    print(
#        "results/mode1.xdmf"
#    )

