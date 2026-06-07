import numpy as np

from petsc4py import PETSc

from dolfinx import fem, mesh


def create_clamp_bc(
    solid_mesh,
    V,
    x_clamp=0.25
):
    """
    Encastrement du bord gauche.
    """

    fdim = solid_mesh.topology.dim - 1

    left_facets = mesh.locate_entities_boundary(
        solid_mesh,
        fdim,
        lambda x: np.isclose(
            x[0],
            x_clamp
        )
    )

    clamp_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        left_facets
    )

    u_D = np.array(
        (0.0, 0.0),
        dtype=PETSc.ScalarType
    )

    bc = fem.dirichletbc(
        u_D,
        clamp_dofs,
        V
    )

    return bc, left_facets
