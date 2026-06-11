import numpy as np

from dolfinx import fem, io

from common.mesh_utils import (
    load_turek_mesh,
    extract_solid_mesh,
    extract_fluid_mesh
)

from common.boundary_transfer import (
    transfer_facet_tags_to_submesh
)

from common.interface_transfer import (
    get_interface_nodes,
    build_interface_mapping,
    extract_interface_values,
    transfer_interface_values
)

from common.boundary_conditions import (
    create_ale_bcs_from_values
)

from common.ale import (
    create_ale_space,
    solve_ale
)

# ======================================
# Lecture maillage
# ======================================

domain, cell_tags, facet_tags = \
    load_turek_mesh()

# ======================================
# Structure
# ======================================

solid_mesh, solid_cell_map, _, _ = \
    extract_solid_mesh(
        domain,
        cell_tags
    )

solid_facet_tags = \
    transfer_facet_tags_to_submesh(
        domain,
        facet_tags,
        solid_mesh,
        solid_cell_map
    )

Vs = fem.functionspace(
    solid_mesh,
    ("Lagrange", 1, (2,))
)

us = fem.Function(Vs)

us.interpolate(
    lambda x: (
        np.zeros_like(x[0]),
        0.01*np.sin(
            np.pi*(x[0]-0.25)/0.35
        )
    )
)

solid_dofs, solid_coords = \
    get_interface_nodes(
        solid_mesh,
        solid_facet_tags,
        Vs
    )

solid_values = \
    extract_interface_values(
        us,
        solid_dofs
    )

# ======================================
# Fluide
# ======================================

fluid_mesh, fluid_cell_map, _, _ = \
    extract_fluid_mesh(
        domain,
        cell_tags
    )

fluid_facet_tags = \
    transfer_facet_tags_to_submesh(
        domain,
        facet_tags,
        fluid_mesh,
        fluid_cell_map
    )

Vf = create_ale_space(
    fluid_mesh
)

fluid_dofs, fluid_coords = \
    get_interface_nodes(
        fluid_mesh,
        fluid_facet_tags,
        Vf
    )

mapping = build_interface_mapping(
    solid_coords,
    fluid_coords
)

fluid_values = \
    transfer_interface_values(
        solid_values,
        mapping
    )

bcs = create_ale_bcs_from_values(
    fluid_mesh,
    fluid_facet_tags,
    Vf,
    fluid_values
)

d = solve_ale(
    Vf,
    bcs
)

print()
print(
    "Max ALE displacement =",
    np.max(
        np.abs(
            d.x.array
        )
    )
)

xdmf = io.XDMFFile(
    fluid_mesh.comm,
    "results/structure_driven_ale.xdmf",
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
    "Saved: results/structure_driven_ale.xdmf"
)

