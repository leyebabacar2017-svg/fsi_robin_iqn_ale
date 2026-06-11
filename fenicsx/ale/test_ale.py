from dolfinx import io

from common.mesh_utils import (
    load_turek_mesh,
    extract_fluid_mesh
)

from common.boundary_transfer import (
    transfer_facet_tags_to_submesh
)

from common.boundary_conditions import (
    create_ale_bcs
)

from common.ale import (
    create_ale_space,
    solve_ale
)

from common.io_utils import (
    print_mesh_info
)

# =====================================
# Lecture maillage
# =====================================

domain, cell_tags, facet_tags = \
    load_turek_mesh()

fluid_mesh, cell_map, _, _ = \
    extract_fluid_mesh(
        domain,
        cell_tags
    )

print_mesh_info(
    fluid_mesh,
    "ALE test"
)

# =====================================
# Tags
# =====================================

fluid_facet_tags = \
    transfer_facet_tags_to_submesh(
        domain,
        facet_tags,
        fluid_mesh,
        cell_map
    )

# =====================================
# ALE
# =====================================

V = create_ale_space(
    fluid_mesh
)

bcs = create_ale_bcs(
    fluid_mesh,
    fluid_facet_tags,
    V
)

d = solve_ale(
    V,
    bcs
)

print()
print("ALE solve completed")

print(
    "Max displacement =",
    abs(d.x.array).max()
)

# =====================================
# Export
# =====================================

xdmf = io.XDMFFile(
    fluid_mesh.comm,
    "results/ale_test.xdmf",
    "w"
)

xdmf.write_mesh(
    fluid_mesh
)

xdmf.write_function(
    d
)

xdmf.close()

print()
print(
    "Saved: results/ale_test.xdmf"
)
