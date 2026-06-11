import numpy as np

from dolfinx import fem

from common.tags import Tags


def get_interface_nodes(
    mesh,
    facet_tags,
    V
):
    """
    Récupère les dofs et coordonnées
    de l'interface FSI.
    """

    fdim = mesh.topology.dim - 1

    mesh.topology.create_connectivity(
        fdim,
        mesh.topology.dim
    )

    interface_facets = facet_tags.find(
        Tags.INTERFACE
    )

    interface_dofs = fem.locate_dofs_topological(
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


def build_interface_mapping(
    solid_coords,
    fluid_coords,
    tol=1.0e-12
):
    """
    Mapping géométrique :
        solide -> fluide
    """

    mapping = []

    for xs in solid_coords:

        dist = np.linalg.norm(
            fluid_coords[:, :2] - xs[:2],
            axis=1
        )

        j = np.argmin(dist)

        if dist[j] > tol:

            raise RuntimeError(
                f"No matching node found "
                f"for {xs}"
            )

        mapping.append(j)

    return np.array(
        mapping,
        dtype=np.int32
    )


def transfer_displacement(
    us_values,
    mapping
):
    """
    Transfert du déplacement
    interface structure -> fluide.
    """

    return us_values[mapping]


def extract_interface_values(
    u,
    interface_dofs
):
    """
    Extraction des valeurs du déplacement
    sur l'interface.
    """

    return u.x.array.reshape(
        (-1, 2)
    )[interface_dofs]


def transfer_interface_values(
    solid_values,
    mapping
):
    """
    Structure -> fluide.
    """

    return solid_values[mapping]

