import numpy as np

from petsc4py import PETSc
from dolfinx import fem

from common.tags import Tags


def create_clamp_bc(
    solid_mesh,
    solid_facet_tags,
    V
):
    """
    Encastrement basé sur les tags Gmsh.
    """

    fdim = solid_mesh.topology.dim - 1

    clamp_facets = solid_facet_tags.find(
        Tags.CLAMP
    )
 
    solid_mesh.topology.create_connectivity(
        fdim,
        solid_mesh.topology.dim
    )   
    
    clamp_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        clamp_facets
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

    return bc, clamp_facets