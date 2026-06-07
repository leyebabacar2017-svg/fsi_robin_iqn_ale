from mpi4py import MPI
import numpy as np

from dolfinx.io.gmsh import read_from_msh
from dolfinx import mesh

mesh_data = read_from_msh(
    "../../meshes/fsi/fsi.msh",
    MPI.COMM_WORLD,
    0
)

domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

solid_cells = cell_tags.find(20)

submesh, cell_map, vertex_map, geom_map = \
    mesh.create_submesh(
        domain,
        domain.topology.dim,
        solid_cells
    )

print()
print("Global interface facets :",
      len(facet_tags.find(102)))

print("Global clamp facets :",
      len(facet_tags.find(103)))

print()
print("Submesh cells :",
      submesh.topology.index_map(
          submesh.topology.dim
      ).size_local)

