from mpi4py import MPI
import numpy as np

from dolfinx.io.gmsh import read_from_msh

mesh_data = read_from_msh(
    "../../meshes/fsi/fsi.msh",
    MPI.COMM_WORLD,
    0
)

cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

print()
print("========== TAGS ==========")

print(
    "Cell tags :",
    np.unique(cell_tags.values)
)

print(
    "Facet tags :",
    np.unique(facet_tags.values)
)

fluid_cells = cell_tags.find(10)
solid_cells = cell_tags.find(20)

print()
print("Fluid cells :", len(fluid_cells))
print("Solid cells :", len(solid_cells))

interface_facets = facet_tags.find(102)
clamp_facets = facet_tags.find(103)

print()
print("Interface facets :", len(interface_facets))
print("Clamp facets :", len(clamp_facets))

print("==========================")
