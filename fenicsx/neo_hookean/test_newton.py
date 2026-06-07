from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import ufl
import numpy as np

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
    [10, 10]
)

V = fem.functionspace(domain, ("Lagrange", 1))

u = fem.Function(V)

v = ufl.TestFunction(V)

F = u * v * ufl.dx

problem = NewtonSolverNonlinearProblem(F, u)

solver = NewtonSolver(MPI.COMM_WORLD, problem)

print("NewtonSolver créé avec succès")
