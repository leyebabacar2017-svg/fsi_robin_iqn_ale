import numpy as np

from petsc4py import PETSc

from dolfinx import fem, mesh

import ufl

from common.tags import Tags

def create_interface_boundary(
    solid_mesh,
    solid_facet_tags
):
    """
    Interface fluide-structure.
    Utilise les tags Gmsh.
    """

    fdim = solid_mesh.topology.dim - 1

    interface_facets = \
        solid_facet_tags.find(
            Tags.INTERFACE
        )

    facet_marker = np.full(
        len(interface_facets),
        1,
        dtype=np.int32
    )

    facet_tags = mesh.meshtags(
        solid_mesh,
        fdim,
        interface_facets,
        facet_marker
    )

    ds = ufl.Measure(
        "ds",
        domain=solid_mesh,
        subdomain_data=facet_tags
    )

    return interface_facets, ds


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
