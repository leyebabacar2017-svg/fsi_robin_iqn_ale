from common.mesh_utils import (
    load_turek_mesh,
    extract_fluid_mesh
)

from common.boundary_transfer import (
    transfer_facet_tags_to_submesh
)

from common.io_utils import (
    print_mesh_info
)

from common.tags import Tags


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
    "Fluid mesh"
)

# =====================================
# Transfert tags frontières
# =====================================

fluid_facet_tags = \
    transfer_facet_tags_to_submesh(
        domain,
        facet_tags,
        fluid_mesh,
        cell_map
    )

# =====================================
# Récupération des frontières
# =====================================

interface_facets = \
    fluid_facet_tags.find(
        Tags.INTERFACE
    )

inlet_facets = \
    fluid_facet_tags.find(
        Tags.INLET
    )

outlet_facets = \
    fluid_facet_tags.find(
        Tags.OUTLET
    )

top_facets = \
    fluid_facet_tags.find(
        Tags.TOP_WALL
    )

bottom_facets = \
    fluid_facet_tags.find(
        Tags.BOTTOM_WALL
    )

cylinder_facets = \
    fluid_facet_tags.find(
        Tags.CYLINDER
    )

# =====================================
# Affichage
# =====================================

print()

print(
    "Interface facets =",
    len(interface_facets)
)

print(
    "Inlet facets =",
    len(inlet_facets)
)

print(
    "Outlet facets =",
    len(outlet_facets)
)

print(
    "Top wall facets =",
    len(top_facets)
)

print(
    "Bottom wall facets =",
    len(bottom_facets)
)

print(
    "Cylinder facets =",
    len(cylinder_facets)
)

print()

print(
    "Available tags =",
    sorted(
        set(
            fluid_facet_tags.values
        )
    )
)
