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
def create_ale_bcs(
    fluid_mesh,
    fluid_facet_tags,
    V
):
    """
    BC ALE :

    Interface :
        d = (0, 0.01)

    Inlet :
        d = 0

    Outlet :
        d = 0

    Top wall :
        d = 0

    Bottom wall :
        d = 0

    Cylinder :
        d = 0
    """

    fdim = fluid_mesh.topology.dim - 1

    fluid_mesh.topology.create_connectivity(
        fdim,
        fluid_mesh.topology.dim
    )

    # =================================
    # Interface
    # =================================

    interface_facets = fluid_facet_tags.find(
        Tags.INTERFACE
    )

    interface_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        interface_facets
    )

    d_interface = np.array(
        (0.0, 0.01),
        dtype=PETSc.ScalarType
    )

    bc_interface = fem.dirichletbc(
        d_interface,
        interface_dofs,
        V
    )

    # =================================
    # Frontières fixes
    # =================================

    fixed_facets = np.hstack(
        (
            fluid_facet_tags.find(
                Tags.INLET
            ),
            fluid_facet_tags.find(
                Tags.OUTLET
            ),
            fluid_facet_tags.find(
                Tags.TOP_WALL
            ),
            fluid_facet_tags.find(
                Tags.BOTTOM_WALL
            ),
            fluid_facet_tags.find(
                Tags.CYLINDER
            )
        )
    )

    fixed_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        fixed_facets
    )

    d_zero = np.array(
        (0.0, 0.0),
        dtype=PETSc.ScalarType
    )

    bc_fixed = fem.dirichletbc(
        d_zero,
        fixed_dofs,
        V
    )

    return [
        bc_interface,
        bc_fixed
    ]


def create_ale_bcs_from_structure(
    fluid_mesh,
    fluid_facet_tags,
    V,
    interface_displacement
):
    """
    ALE BC à partir d'un déplacement
    structure déjà connu.
    """

    fdim = fluid_mesh.topology.dim - 1

    fluid_mesh.topology.create_connectivity(
        fdim,
        fluid_mesh.topology.dim
    )

    interface_facets = fluid_facet_tags.find(
        Tags.INTERFACE
    )

    interface_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        interface_facets
    )

    bc_interface = fem.dirichletbc(
        interface_displacement,
        interface_dofs
    )

    fixed_facets = np.hstack(
        (
            fluid_facet_tags.find(Tags.INLET),
            fluid_facet_tags.find(Tags.OUTLET),
            fluid_facet_tags.find(Tags.TOP_WALL),
            fluid_facet_tags.find(Tags.BOTTOM_WALL),
            fluid_facet_tags.find(Tags.CYLINDER)
        )
    )

    fixed_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        fixed_facets
    )

    zero = np.array(
        (0.0, 0.0),
        dtype=PETSc.ScalarType
    )

    bc_fixed = fem.dirichletbc(
        zero,
        fixed_dofs,
        V
    )

    return [
        bc_interface,
        bc_fixed
    ]


def create_ale_bcs_from_values(
    fluid_mesh,
    fluid_facet_tags,
    V,
    fluid_interface_values
):
    """
    ALE BC construite à partir des
    déplacements transférés.
    """

    from dolfinx import fem
    import numpy as np
    from petsc4py import PETSc

    fdim = fluid_mesh.topology.dim - 1

    fluid_mesh.topology.create_connectivity(
        fdim,
        fluid_mesh.topology.dim
    )

    interface_facets = fluid_facet_tags.find(
        Tags.INTERFACE
    )

    interface_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        interface_facets
    )

    d_interface = fem.Function(V)

    values = d_interface.x.array.reshape(
        (-1, 2)
    )

    values[interface_dofs] = fluid_interface_values

    bc_interface = fem.dirichletbc(
        d_interface,
        interface_dofs
    )

    fixed_facets = np.hstack(
        (
            fluid_facet_tags.find(Tags.INLET),
            fluid_facet_tags.find(Tags.OUTLET),
            fluid_facet_tags.find(Tags.TOP_WALL),
            fluid_facet_tags.find(Tags.BOTTOM_WALL),
            fluid_facet_tags.find(Tags.CYLINDER)
        )
    )

    fixed_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        fixed_facets
    )

    zero = np.array(
        (0.0, 0.0),
        dtype=PETSc.ScalarType
    )

    bc_fixed = fem.dirichletbc(
        zero,
        fixed_dofs,
        V
    )

    return [
        bc_interface,
        bc_fixed
    ]

