from mpi4py import MPI
from dolfinx import io, mesh
import numpy as np

# =====================================================
# Lecture du maillage
# =====================================================

with io.XDMFFile(MPI.COMM_WORLD, "solid.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")

# =====================================================
# Lecture des facettes
# =====================================================

with io.XDMFFile(MPI.COMM_WORLD, "solid_facets.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(
        domain,
        name="Grid"
    )

# =====================================================
# Vérification
# =====================================================

interface_facets = facet_tags.find(102)
clamp_facets     = facet_tags.find(103)

print()
print("====================================")
print("Interface facets :", len(interface_facets))
print("Clamp facets     :", len(clamp_facets))
print("====================================")

print()
print("Tags présents :")
print(np.unique(facet_tags.values))

