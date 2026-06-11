import numpy as np

from dolfinx import fem

from common.mesh_utils import (
    load_turek_mesh,
    extract_solid_mesh,
    extract_fluid_mesh
)

from common.boundary_transfer import (
    transfer_facet_tags_to_submesh
)

from common.tags import Tags


def get_interface_nodes(
    mesh,
    facet_tags,
    V
):
    """
    Retourne les dofs et coordonnées
    de l'interface.
    """

    fdim = mesh.topology.dim - 1

    mesh.topology.create_connectivity(
        fdim,
        mesh.topology.dim
    )

    interface_facets = facet_tags.find(
        Tags.INTERFACE
    )

    interface_dofs = \
        fem.locate_dofs_topological(
            V,
            fdim,
            interface_facets
        )

    interface_dofs = np.unique(
        interface_dofs
    )

    coords = V.tabulate_dof_coordinates()

    return (
        interface_dofs,
        coords[interface_dofs]
    )


# =====================================
# Lecture maillage global
# =====================================

domain, cell_tags, facet_tags = \
    load_turek_mesh()

# =====================================
# Structure
# =====================================

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

# =====================================
# Fluide
# =====================================

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

# =====================================
# Affichage
# =====================================

print()
print(
    "Structure interface nodes =",
    len(solid_dofs)
)

print(
    "Fluid interface nodes =",
    len(fluid_dofs)
)

print()

print("First structure coordinates")

for x in solid_coords[:10]:

    print(x)

print()

print("First fluid coordinates")

for x in fluid_coords[:10]:

    print(x)

