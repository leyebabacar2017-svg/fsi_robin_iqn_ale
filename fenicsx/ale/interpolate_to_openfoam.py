import numpy as np

# =====================================
# Interface FEniCS
# =====================================

data = np.loadtxt(
    "fenicsx/results/interface_history/interface_01000.csv",
    skiprows=1
)

y_fem  = data[:,1]
ux_fem = data[:,2]
uy_fem = data[:,3]

# =====================================
# Interface OpenFOAM
# =====================================

y_of = np.array([
0.19253957,
0.19375585,
0.19500039,
0.19625000,
0.19750000,
0.19875000,
0.20000000,
0.20125000,
0.20250000,
0.20375000,
0.20499961,
0.20624415,
0.20746043
])

ux_of = np.interp(
    y_of,
    y_fem,
    ux_fem
)

uy_of = np.interp(
    y_of,
    y_fem,
    uy_fem
)

print()
print("OpenFOAM interface displacement")

for y, ux, uy in zip(
    y_of,
    ux_of,
    uy_of
):
    print(
        f"y={y:.6f}"
        f" ux={ux:.6e}"
        f" uy={uy:.6e}"
    )

np.savetxt(
    "openfoam_interface_displacement.csv",
    np.column_stack(
        [
            y_of,
            ux_of,
            uy_of
        ]
    ),
    header="y ux uy"
)

print()
print(
    "Saved openfoam_interface_displacement.csv"
)
