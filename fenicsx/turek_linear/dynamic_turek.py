from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl

from dolfinx import fem, io
from common.tags import Tags

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

from common.time_integrators import (
    build_newmark_matrix,
    create_linear_solver
)

from common.io_utils import (
    print_mesh_info
)

from common.parameters import (
    TurekParameters
)

from common.probes import (
    find_tip_dof
)

from dolfinx.fem.petsc import (
    assemble_matrix
)

from common.loads import (
    assemble_load_vector
)

from common.traction import (
    create_interface_boundary,
    create_vertical_traction,
    create_interface_force_form
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
    
solid_facet_tags = \
    transfer_facet_tags_to_submesh(
        domain,
        facet_tags,
        solid_mesh,
        cell_map
    )
    
print_mesh_info(
    solid_mesh,
    "Dynamic Turek"
)

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

rho = TurekParameters.rho

E = TurekParameters.E

nu = TurekParameters.nu

mu, lmbda = lame_parameters(
    E,
    nu
)

# =====================================================
# BC
# =====================================================

bc, clamp_facets = \
    create_clamp_bc(
        solid_mesh,
        solid_facet_tags,
        V
    )

interface_facets, ds_interface = \
    create_interface_boundary(
        solid_mesh,
        solid_facet_tags
    )

print()
print(
    "Interface facets =",
    len(interface_facets)
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

m_form = rho*ufl.inner(u,v)*dx

k_form = ufl.inner(
    sigma(
        u,
        mu,
        lmbda
    ),
    eps(v)
)*dx

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
print("Mass matrix size:")
print(M.getSize())

print()
print("Stiffness matrix size:")
print(K.getSize())

print()
print(
    "||M||F =",
    M.norm()
)

print(
    "||K||F =",
    K.norm()
)

# =====================================================
# Newmark
# =====================================================

T = TurekParameters.T

dt = TurekParameters.dt

beta = TurekParameters.beta

gamma = TurekParameters.gamma

Keff = build_newmark_matrix(
    M,
    K,
    beta,
    dt
)

solver = create_linear_solver(
    Keff,
    solid_mesh.comm
)

traction = create_vertical_traction(
    solid_mesh,
    TurekParameters.traction
)

load_form = \
    create_interface_force_form(
        traction,
        v,
        ds_interface
    )

#F = assemble_load_vector(
#    load_form,
#    [bc]
#)

F = assemble_load_vector(
    load_form
)
print()
print(
    "||F||2 =",
    F.norm()
)
nsteps = int(T/dt)

uh = fem.Function(V)
vh = fem.Function(V)
ah = fem.Function(V)

u_old = fem.Function(V)
v_old = fem.Function(V)
a_old = fem.Function(V)

# impulsion initiale
vh.x.array[:] = 0.0
ah.x.array[:] = 0.0

coords = V.tabulate_dof_coordinates()

tip_dof = find_tip_dof(
    V
)

print()
print(
    "Tip dof =",
    tip_dof
)

print()

print(
    "Tip coordinates =",
    coords[tip_dof]
)

time_history = []
tip_history = []

U = uh.x.petsc_vec

rhs = U.duplicate()

tmp = U.duplicate()

vh.x.array[:] = 0.0
ah.x.array[:] = 0.0

u_old.x.array[:] = uh.x.array[:]
v_old.x.array[:] = vh.x.array[:]
a_old.x.array[:] = ah.x.array[:]
  
# =====================================================
# Export
# =====================================================

xdmf = io.XDMFFile(
    solid_mesh.comm,
    "results/dynamic_turek.xdmf",
    "w"
)

xdmf.write_mesh(solid_mesh)

print()
print("Starting time loop")

for n in range(nsteps):

    t = (n+1)*dt

    if n % 100 == 0:

        print(
            f"step {n}/{nsteps}"
        )

    # --------------------------------
    # RHS Newmark
    # --------------------------------

    tmp.array[:] = (
        1.0/(beta*dt*dt)
        * u_old.x.array
        +
        1.0/(beta*dt)
        * v_old.x.array
        +
        (
            1.0/(2.0*beta)-1.0
        )
        * a_old.x.array
    )

    rhs.zeroEntries()

    M.mult(
        tmp,
        rhs
    )

    rhs.axpy(
       1.0,
       F
    )  
    
    # --------------------------------
    # Solve
    # --------------------------------

    solver.solve(
        rhs,
        U
    )

    uh.x.scatter_forward()

    # --------------------------------
    # Acceleration
    # --------------------------------

    ah.x.array[:] = (
        1.0/(beta*dt*dt)
        * (
            uh.x.array
            -
            u_old.x.array
        )
        -
        1.0/(beta*dt)
        * v_old.x.array
        -
        (
            1.0/(2.0*beta)-1.0
        )
        * a_old.x.array
    )

    # --------------------------------
    # Velocity
    # --------------------------------

    vh.x.array[:] = (
        v_old.x.array
        +
        dt
        * (
            (1.0-gamma)
            * a_old.x.array
            +
            gamma
            * ah.x.array
        )
    )

    # --------------------------------
    # Update
    # --------------------------------

    u_old.x.array[:] = uh.x.array[:]
    v_old.x.array[:] = vh.x.array[:]
    a_old.x.array[:] = ah.x.array[:]

    # --------------------------------
    # Export
    # --------------------------------
    time_history.append(t)

    tip_history.append(
        uh.x.array[
            2*tip_dof + 1
        ]
    )
    
    xdmf.write_function(
        uh,
        t
    )

xdmf.close()

print()
print(
    "Tip displacement =",
    tip_history[-1]
)

print()
print(
    "Maximum tip displacement =",
    np.max(
        np.abs(
            np.array(tip_history)
        )
    )
)

data = np.column_stack(
    (
        time_history,
        tip_history
    )
)

np.savetxt(
    "results/tip_history.txt",
    data,
    header="time uy"
)

print()
print("Min tip =", np.min(tip_history))
print("Max tip =", np.max(tip_history))

print()
print("Finished")
