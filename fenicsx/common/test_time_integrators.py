import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from petsc4py import PETSc

from common.time_integrators import (
    create_linear_solver
)

A = PETSc.Mat().createAIJ([2,2])

A.setUp()

A.setValue(0,0,2.0)
A.setValue(1,1,3.0)

A.assemble()

solver = create_linear_solver(
    A,
    PETSc.COMM_WORLD
)

print()
print("Solver OK")