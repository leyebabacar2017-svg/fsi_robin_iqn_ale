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
    
print()
print("====================================")
print("Dynamic Turek")
print(
    "Cells =",
    solid_mesh.topology.index_map(
        solid_mesh.topology.dim
    ).size_local
)
print(
    "Nodes =",
    solid_mesh.geometry.x.shape[0]
)
print("====================================")

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

rho = 1000.0

E = TurekParameters.E

nu = TurekParameters.nu

mu, lmbda = lame_parameters(
    E,
    nu_m
)

# =====================================================
# BC
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

# =====================================================
# Newmark
# =====================================================

TRACTION = TurekParameters.traction

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

U = uh.x.petsc_vec
Vv = vh.x.petsc_vec
Aa = ah.x.petsc_vec

rhs = U.duplicate()

tmp = U.duplicate()

for i in range(len(coords)):

    x = coords[i]

    if abs(x[0]-0.60) < 1e-3:

        vh.x.array[2*i+1] = -0.1
  
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

    xdmf.write_function(
        uh,
        t
    )

xdmf.close()

print()
print("Finished")
