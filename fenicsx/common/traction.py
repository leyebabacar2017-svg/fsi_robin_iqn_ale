from dolfinx import mesh
from dolfinx import fem

from petsc4py import PETSc

import numpy as np
import ufl

from common.tags import Tags


def create_interface_boundary(
    solid_mesh,
    solid_facet_tags
):
    """
    Interface solide-fluide basée
    sur les tags Gmsh.
    """

    fdim = solid_mesh.topology.dim - 1

    interface_facets = solid_facet_tags.find(
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


def create_interface_force_form(
    traction,
    v,
    ds_interface
):
    """
    Forme linéaire associée
    à une traction imposée.
    """

    return (
        ufl.dot(
            traction,
            v
        )
        * ds_interface(1)
    )


def create_vertical_traction(
    solid_mesh,
    value
):
    """
    Traction verticale uniforme.
    """

    return fem.Constant(
        solid_mesh,
        PETSc.ScalarType(
            (0.0, -value)
        )
    )