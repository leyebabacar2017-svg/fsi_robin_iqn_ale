from petsc4py import PETSc


def build_newmark_matrix(
    M,
    K,
    beta,
    dt
):
    """
    Construction de

    Keff = K + M/(beta*dt²)
    """

    Keff = K.copy()

    Keff.axpy(
        1.0/(beta*dt*dt),
        M
    )

    Keff.assemble()

    return Keff


def create_linear_solver(
    A,
    comm
):
    """
    Solveur direct PETSc.
    """

    solver = PETSc.KSP().create(
        comm
    )

    solver.setOperators(A)

    solver.setType(
        "preonly"
    )

    solver.getPC().setType(
        "lu"
    )

    return solver
