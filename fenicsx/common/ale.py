from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

import ufl


def create_ale_space(
    fluid_mesh
):
    """
    Espace EF vectoriel pour ALE.
    """

    return fem.functionspace(
        fluid_mesh,
        ("Lagrange", 1, (2,))
    )


def build_ale_forms(
    V
):
    """
    Résolution de

        -Delta(d) = 0

    sur le domaine fluide.
    """

    d = ufl.TrialFunction(V)

    w = ufl.TestFunction(V)

    a = ufl.inner(
        ufl.grad(d),
        ufl.grad(w)
    ) * ufl.dx

    zero = fem.Constant(
        V.mesh,
        (0.0, 0.0)
    )

    L = ufl.inner(
        zero,
        w
    ) * ufl.dx

    return a, L


def solve_ale(
    V,
    bcs
):
    """
    Résolution du problème ALE.
    """

    a, L = build_ale_forms(V)

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="ale_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-10
        }
    )

    d = problem.solve()

    return d

