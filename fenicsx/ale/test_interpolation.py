from dolfinx import fem

from common.mesh_utils import (
    load_turek_mesh,
    extract_solid_mesh,
    extract_fluid_mesh
)

domain, cell_tags, facet_tags = \
    load_turek_mesh()

solid_mesh, _, _, _ = \
    extract_solid_mesh(
        domain,
        cell_tags
    )

fluid_mesh, _, _, _ = \
    extract_fluid_mesh(
        domain,
        cell_tags
    )

Vs = fem.functionspace(
    solid_mesh,
    ("Lagrange", 1, (2,))
)

Vf = fem.functionspace(
    fluid_mesh,
    ("Lagrange", 1, (2,))
)

us = fem.Function(Vs)

# déplacement artificiel
us.interpolate(
    lambda x: (
        0*x[0],
        0.01*x[0]
    )
)

print()
print(
    "Max structure displacement =",
    abs(us.x.array).max()
)

uf = fem.Function(Vf)

uf.interpolate(us)

print(
    "Max fluid displacement =",
    abs(uf.x.array).max()
)

print()
print("Interpolation successful")

