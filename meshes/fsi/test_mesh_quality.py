from mpi4py import MPI
import numpy as np

from dolfinx.io.gmsh import read_from_msh

mesh_data = read_from_msh(
    "../../meshes/fsi/fsi.msh",
    MPI.COMM_WORLD,
    0
)

mesh = mesh_data.mesh

tdim = mesh.topology.dim

mesh.topology.create_connectivity(
    tdim,
    0
)

cells = mesh.topology.connectivity(
    tdim,
    0
)

x = mesh.geometry.x

areas = []

n_cells = mesh.topology.index_map(
    tdim
).size_local

for c in range(n_cells):

    verts = cells.links(c)

    p = x[verts]

    x1, y1 = p[0, :2]
    x2, y2 = p[1, :2]
    x3, y3 = p[2, :2]

    area = 0.5 * abs(
        (x2 - x1) * (y3 - y1)
        -
        (x3 - x1) * (y2 - y1)
    )

    areas.append(area)

areas = np.array(areas)

print()
print("Cells     :", len(areas))
print("Min area  :", areas.min())
print("Max area  :", areas.max())
print("Mean area :", areas.mean())
