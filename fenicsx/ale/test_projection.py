from common.mesh_utils import (
    load_turek_mesh,
    extract_solid_mesh,
    extract_fluid_mesh
)

from common.io_utils import (
    print_mesh_info
)

domain, cell_tags, facet_tags = \
    load_turek_mesh()

solid_mesh, _, _, _ = \
    extract_solid_mesh(
        domain,
        cell_tags
    )

fluid_mesh, _, _, _ = \
    extract_fluid_mesh(
        domain,
        cell_tags
    )

print_mesh_info(
    solid_mesh,
    "Solid"
)

print_mesh_info(
    fluid_mesh,
    "Fluid"
)

