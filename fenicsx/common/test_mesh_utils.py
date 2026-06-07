import sys

sys.path.append("..")

from common.mesh_utils import (
    load_turek_mesh,
    extract_solid_mesh
)

domain, cell_tags, facet_tags = \
    load_turek_mesh()

solid_mesh, _, _, _ = \
    extract_solid_mesh(
        domain,
        cell_tags
    )

print()
print("Global cells =",
      domain.topology.index_map(
          domain.topology.dim
      ).size_local)

print("Solid cells =",
      solid_mesh.topology.index_map(
          solid_mesh.topology.dim
      ).size_local)

print("Solid nodes =",
      solid_mesh.geometry.x.shape[0])
