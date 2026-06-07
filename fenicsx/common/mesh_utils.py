from pathlib import Path
from mpi4py import MPI

from dolfinx.io.gmsh import read_from_msh
from dolfinx import mesh

from common.tags import Tags

def load_turek_mesh(msh_file=None):

    if msh_file is None:

        root = Path(__file__).resolve().parents[2]

        msh_file = (
            root
            / "meshes"
            / "fsi"
            / "fsi.msh"
        )

    mesh_data = read_from_msh(
        str(msh_file),
        MPI.COMM_WORLD,
        0
    )

    return (
        mesh_data.mesh,
        mesh_data.cell_tags,
        mesh_data.facet_tags
    )

def extract_solid_mesh(
    domain,
    cell_tags
):
    """
    Extraction du sous-maillage solide.
    """

    solid_cells = cell_tags.find(
        Tags.SOLID
    )

    return mesh.create_submesh(
        domain,
        domain.topology.dim,
        solid_cells
    )


def extract_fluid_mesh(
    domain,
    cell_tags
):
    """
    Extraction du sous-maillage fluide.
    """

    fluid_cells = cell_tags.find(
        Tags.FLUID
    )

    return mesh.create_submesh(
        domain,
        domain.topology.dim,
        fluid_cells
    )
