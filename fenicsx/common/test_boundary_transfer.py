import sys

sys.path.append("..")

from common.tags import Tags
from common.mesh_utils import (
    load_turek_mesh,
    extract_solid_mesh
)

from common.boundary_transfer import (
    transfer_facet_tags_to_submesh
)

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

print()
print(
    "Clamp facets =",
    len(
        solid_facet_tags.find(
            Tags.CLAMP
        )
    )
)

print(
    "Interface facets =",
    len(
        solid_facet_tags.find(
            Tags.INTERFACE
        )
    )
)
