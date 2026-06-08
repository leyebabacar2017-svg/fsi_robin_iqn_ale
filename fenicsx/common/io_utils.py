def print_banner(
    title
):
    """
    Affichage standard.
    """

    print()
    print(
        "===================================="
    )

    print(title)

    print(
        "===================================="
    )


def print_mesh_info(
    mesh,
    title="Mesh"
):
    """
    Informations maillage.
    """

    print()
    print(
        "===================================="
    )

    print(title)

    print(
        "Cells :",
        mesh.topology.index_map(
            mesh.topology.dim
        ).size_local
    )

    print(
        "Nodes :",
        mesh.geometry.x.shape[0]
    )

    print(
        "===================================="
    )

    print()
