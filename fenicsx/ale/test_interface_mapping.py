from dolfinx import fem

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
    build_interface_mapping
)

from common.tags import Tags

domain, cell_tags, facet_tags = \
    load_turek_mesh()

# ==================================
# Structure
# ==================================

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

solid_dofs, solid_coords = \
    get_interface_nodes(
        solid_mesh,
        solid_facet_tags,
        Vs
    )

# ==================================
# Fluide
# ==================================

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

Vf = fem.functionspace(
    fluid_mesh,
    ("Lagrange", 1, (2,))
)

fluid_dofs, fluid_coords = \
    get_interface_nodes(
        fluid_mesh,
        fluid_facet_tags,
        Vf
    )

# ==================================
# Mapping
# ==================================

mapping = build_interface_mapping(
    solid_coords,
    fluid_coords
)

print()
print(
    "Interface nodes =",
    len(mapping)
)

print(
    "First mappings:"
)

for i in range(10):

    print(
        i,
        "->",
        mapping[i]
    )

print()
print("Mapping successful")

