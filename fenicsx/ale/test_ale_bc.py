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
    create_ale_space
)

domain, cell_tags, facet_tags = \
    load_turek_mesh()

fluid_mesh, cell_map, _, _ = \
    extract_fluid_mesh(
        domain,
        cell_tags
    )

fluid_facet_tags = \
    transfer_facet_tags_to_submesh(
        domain,
        facet_tags,
        fluid_mesh,
        cell_map
    )

V = create_ale_space(
    fluid_mesh
)

bcs = create_ale_bcs(
    fluid_mesh,
    fluid_facet_tags,
    V
)

print()
print(
    "Number of BCs =",
    len(bcs)
)

print()
print("ALE BC creation successful")
