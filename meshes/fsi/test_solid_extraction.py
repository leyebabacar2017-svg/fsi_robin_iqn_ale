from mpi4py import MPI

from dolfinx.io.gmsh import read_from_msh
from dolfinx import mesh

mesh_data = read_from_msh(
    "../../meshes/fsi/fsi.msh",
    MPI.COMM_WORLD,
    0
)

domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags

solid_cells = cell_tags.find(20)

print()
print("Solid cells =", len(solid_cells))

solid_mesh, cell_map, vertex_map, geom_map = \
    mesh.create_submesh(
        domain,
        domain.topology.dim,
        solid_cells
    )

print()
print("Submesh created")

print(
    "Cells =",
    solid_mesh.topology.index_map(
        solid_mesh.topology.dim
    ).size_local
)

print(
    "Nodes =",
    solid_mesh.geometry.x.shape[0]
)
