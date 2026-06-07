from mpi4py import MPI
import numpy as np

from dolfinx.io.gmsh import read_from_msh

mesh_data = read_from_msh(
    "../../meshes/fsi/fsi.msh",
    MPI.COMM_WORLD,
    0
)

domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

print()
print("========== TAGS ==========")

print("Cell tags :",
      np.unique(cell_tags.values))

print("Facet tags :",
      np.unique(facet_tags.values))

print()

print("Fluid cells :",
      len(cell_tags.find(10)))

print("Solid cells :",
      len(cell_tags.find(20)))

print()

print("Interface facets :",
      len(facet_tags.find(102)))

print("Clamp facets :",
      len(facet_tags.find(103)))

print("==========================")

