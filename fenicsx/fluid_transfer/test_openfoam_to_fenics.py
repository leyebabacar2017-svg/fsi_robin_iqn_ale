from fluid_transfer.read_openfoam_force import (
    read_openfoam_force
)

traction = read_openfoam_force(
    "openfoam/turek_cylinderA_validated/"
    "postProcessing/forcesBeam/0/force.dat"
)

print()
print("Traction transferred to FEniCS")
print(traction)
