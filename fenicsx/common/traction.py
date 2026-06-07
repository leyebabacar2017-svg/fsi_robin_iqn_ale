import numpy as np

from petsc4py import PETSc

from dolfinx import fem, mesh

import ufl


def create_right_boundary(
    solid_mesh,
    x_right=0.60
):
    """
    Détection du bord libre.
    """

    fdim = solid_mesh.topology.dim - 1

    right_facets = mesh.locate_entities_boundary(
        solid_mesh,
        fdim,
        lambda x: np.isclose(
            x[0],
            x_right
        )
    )

    facet_marker = np.full(
        len(right_facets),
        1,
        dtype=np.int32
    )

    facet_tags = mesh.meshtags(
        solid_mesh,
        fdim,
        right_facets,
        facet_marker
    )

    ds = ufl.Measure(
        "ds",
        domain=solid_mesh,
        subdomain_data=facet_tags
    )

    return right_facets, ds


def create_vertical_traction(
    solid_mesh,
    value
):
    """
    Traction verticale.
    """

    return fem.Constant(
        solid_mesh,
        PETSc.ScalarType(
            (0.0, -value)
        )
    )
