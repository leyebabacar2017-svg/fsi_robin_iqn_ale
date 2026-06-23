import numpy as np
import re

# =====================================
# Nombre total de points OpenFOAM
# =====================================

n_points = 32987

disp = np.zeros(
    (n_points, 3)
)

# =====================================
# Points interface
# =====================================

ids = np.loadtxt(
    "openfoam_interface_ids.csv"
)

# =====================================
# Déplacements interpolés
# =====================================

data = np.loadtxt(
    "openfoam_interface_displacement.csv",
    skiprows=1
)

y_of  = data[:,0]
ux_of = data[:,1]
uy_of = data[:,2]

# =====================================
# Affectation
# =====================================

for row in ids:

    pid = int(row[0])

    y = row[2]

    j = np.argmin(
        np.abs(y_of - y)
    )

    disp[pid,0] = ux_of[j]
    disp[pid,1] = uy_of[j]

# =====================================
# Ecriture OpenFOAM
# =====================================

with open(
    "pointDisplacement",
    "w"
) as f:

    f.write(
        "FoamFile\n"
        "{\n"
        "    version 2.0;\n"
        "    format ascii;\n"
        "    class vectorField;\n"
        "    object pointDisplacement;\n"
        "}\n\n"
    )

    f.write(
        str(n_points) + "\n(\n"
    )

    for d in disp:

        f.write(
            f"({d[0]} {d[1]} {d[2]})\n"
        )

    f.write(")\n")

print()
print(
    "Generated pointDisplacement"
)

print(
    "Nonzero points =",
    np.sum(
        np.linalg.norm(
            disp,
            axis=1
        ) > 0
    )
)
