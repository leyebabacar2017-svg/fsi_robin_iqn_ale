from pathlib import Path
import re
import numpy as np

case_dir = Path(
    "openfoam/turek_cylinderA_validated"
)

boundary_file = (
    case_dir /
    "constant/polyMesh/boundary"
)

faces_file = (
    case_dir /
    "constant/polyMesh/faces"
)

points_file = (
    case_dir /
    "constant/polyMesh/points"
)

# ======================================
# Lecture boundary
# ======================================

with open(boundary_file) as f:
    txt = f.read()

m = re.search(
    r"beam\s*\{.*?nFaces\s+(\d+);.*?startFace\s+(\d+);",
    txt,
    re.S
)

nFaces = int(m.group(1))
startFace = int(m.group(2))

print()
print("Beam patch")
print("nFaces =", nFaces)
print("startFace =", startFace)

# ======================================
# Lecture faces
# ======================================

with open(faces_file) as f:
    lines = f.readlines()

face_lines = []

for line in lines:

    if "(" in line and ")" in line:

        face_lines.append(
            line.strip()
        )

beam_faces = face_lines[
    startFace :
    startFace + nFaces
]

beam_points = set()

for face in beam_faces:

    ids = re.findall(
        r"\d+",
        face
    )[1:]

    for p in ids:

        beam_points.add(
            int(p)
        )

beam_points = np.array(
    sorted(beam_points)
)

print()
print(
    "Unique beam points =",
    len(beam_points)
)

# ======================================
# Lecture points
# ======================================

with open(points_file) as f:
    lines = f.readlines()

point_lines = []

for line in lines:

    if line.startswith("(") and ")" in line:

        point_lines.append(
            line.strip()
        )

coords = []

for p in beam_points:

    xyz = re.findall(
        r"[-+eE0-9\.]+",
        point_lines[p]
    )

    coords.append(
        [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2])
        ]
    )

coords = np.array(coords)

print()
print(
    "Beam coordinates shape =",
    coords.shape
)

print()
print(
    coords[:10]
)

np.savetxt(
    "beam_points.csv",
    coords,
    header="x y z"
)

print()
print(
    "Saved beam_points.csv"
)
