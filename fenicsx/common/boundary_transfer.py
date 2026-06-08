import numpy as np

from dolfinx import mesh


def transfer_facet_tags_to_submesh(
    domain,
    facet_tags,
    submesh,
    cell_map
):
    """
    Transfert des tags de frontière du maillage global
    vers un sous-maillage.

    Parameters
    ----------
    domain : Mesh
        Maillage global.

    facet_tags : MeshTags
        Tags du maillage global.

    submesh : Mesh
        Sous-maillage.

    cell_map : EntityMap
        Mapping cellules sous-maillage -> maillage global.

    Returns
    -------
    MeshTags
        Tags sur le sous-maillage.
    """

    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(
        tdim,
        fdim
    )

    submesh.topology.create_connectivity(
        tdim,
        fdim
    )

    global_facets = []
    global_values = []

    tagged_facets = facet_tags.indices
    tagged_values = facet_tags.values

    facet_dict = {
        int(f): int(v)
        for f, v in zip(
            tagged_facets,
            tagged_values
        )
    }

    for local_cell in range(
        submesh.topology.index_map(
            tdim
        ).size_local
    ):

        global_cell = \
            cell_map.sub_topology_to_topology(
                np.array(
                    [local_cell],
                    dtype=np.int32
                ),
                False
            )[0]

        gfacets = domain.topology.connectivity(
            tdim,
            fdim
        ).links(global_cell)

        lfacets = submesh.topology.connectivity(
            tdim,
            fdim
        ).links(local_cell)

        for gf, lf in zip(
            gfacets,
            lfacets
        ):

            if gf in facet_dict:

                global_facets.append(
                    int(lf)
                )

                global_values.append(
                    facet_dict[gf]
                )

    global_facets = np.array(
        global_facets,
        dtype=np.int32
    )

    global_values = np.array(
        global_values,
        dtype=np.int32
    )

    order = np.argsort(
        global_facets
    )

    return mesh.meshtags(
        submesh,
        fdim,
        global_facets[order],
        global_values[order]
    )
